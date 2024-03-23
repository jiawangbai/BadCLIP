import os.path as osp
from collections import OrderedDict
import math

import torch
import torch.nn as nn
from torch.nn import functional as F
from torch.cuda.amp import GradScaler, autocast

from dassl.engine import TRAINER_REGISTRY, TrainerX
from dassl.metrics import compute_accuracy
from dassl.utils import load_pretrained_weights, load_checkpoint, mkdir_if_missing, save_checkpoint, MetricMeter, AverageMeter
from dassl.optim import build_optimizer, build_lr_scheduler

from clip import clip
from clip.simple_tokenizer import SimpleTokenizer as _Tokenizer
import time
import datetime
from tqdm import tqdm

_tokenizer = _Tokenizer()


def load_clip_to_cpu(cfg):
    backbone_name = cfg.MODEL.BACKBONE.NAME
    url = clip._MODELS[backbone_name]
    model_path = clip._download(url)
    
    try:
        # loading JIT archive
        model = torch.jit.load(model_path, map_location="cpu").eval()
        state_dict = None

    except RuntimeError:
        state_dict = torch.load(model_path, map_location="cpu")

    model = clip.build_model(state_dict or model.state_dict())

    return model


class TextEncoder(nn.Module):
    def __init__(self, clip_model):
        super().__init__()
        self.transformer = clip_model.transformer
        self.positional_embedding = clip_model.positional_embedding
        self.ln_final = clip_model.ln_final
        self.text_projection = clip_model.text_projection
        self.dtype = clip_model.dtype

    def forward(self, prompts, tokenized_prompts):
        x = prompts + self.positional_embedding.type(self.dtype)
        x = x.permute(1, 0, 2)  # NLD -> LND
        x = self.transformer(x)
        x = x.permute(1, 0, 2)  # LND -> NLD
        x = self.ln_final(x).type(self.dtype)

        # x.shape = [batch_size, n_ctx, transformer.width]
        # take features from the eot embedding (eot_token is the highest number in each sequence)
        x = x[torch.arange(x.shape[0]), tokenized_prompts.argmax(dim=-1)] @ self.text_projection

        return x


class PromptLearner(nn.Module):
    def __init__(self, cfg, classnames, clip_model):
        super().__init__()
        n_cls = len(classnames)
        n_ctx = cfg.TRAINER.COCOOP.N_CTX
        ctx_init = cfg.TRAINER.COCOOP.CTX_INIT
        dtype = clip_model.dtype
        ctx_dim = clip_model.ln_final.weight.shape[0]
        vis_dim = clip_model.visual.output_dim

        clip_imsize = clip_model.visual.input_resolution
        cfg_imsize = cfg.INPUT.SIZE[0]
        assert cfg_imsize == clip_imsize, f"cfg_imsize ({cfg_imsize}) must equal to clip_imsize ({clip_imsize})"

        if ctx_init:
            # use given words to initialize context vectors
            ctx_init = ctx_init.replace("_", " ")
            n_ctx = len(ctx_init.split(" "))
            prompt = clip.tokenize(ctx_init)
            with torch.no_grad():
                embedding = clip_model.token_embedding(prompt).type(dtype)
            ctx_vectors = embedding[0, 1 : 1 + n_ctx, :]
            prompt_prefix = ctx_init
        else:
            # random initialization
            ctx_vectors = torch.empty(n_ctx, ctx_dim, dtype=dtype)
            nn.init.normal_(ctx_vectors, std=0.02)
            prompt_prefix = " ".join(["X"] * n_ctx)

        print(f'Initial context: "{prompt_prefix}"')
        print(f"Number of context words (tokens): {n_ctx}")

        self.ctx = nn.Parameter(ctx_vectors)

        self.meta_net = nn.Sequential(OrderedDict([
            ("linear1", nn.Linear(vis_dim, vis_dim // 16)),
            ("relu", nn.ReLU(inplace=True)),
            ("linear2", nn.Linear(vis_dim // 16, ctx_dim))
        ]))
        
        if cfg.TRAINER.COCOOP.PREC == "fp16":
            self.meta_net.half()

        classnames = [name.replace("_", " ") for name in classnames]
        name_lens = [len(_tokenizer.encode(name)) for name in classnames]
        prompts = [prompt_prefix + " " + name + "." for name in classnames]

        tokenized_prompts = torch.cat([clip.tokenize(p) for p in prompts])  # (n_cls, n_tkn)
        with torch.no_grad():
            embedding = clip_model.token_embedding(tokenized_prompts).type(dtype)

        # These token vectors will be saved when in save_model(),
        # but they should be ignored in load_model() as we want to use
        # those computed using the current class names
        self.register_buffer("token_prefix", embedding[:, :1, :])  # SOS
        self.register_buffer("token_suffix", embedding[:, 1 + n_ctx :, :])  # CLS, EOS

        self.n_cls = n_cls
        self.n_ctx = n_ctx
        self.tokenized_prompts = tokenized_prompts  # torch.Tensor
        self.name_lens = name_lens
        self.classnames = classnames
    
    def construct_prompts(self, ctx, prefix, suffix, label=None):
        # dim0 is either batch_size (during training) or n_cls (during testing)
        # ctx: context tokens, with shape of (dim0, n_ctx, ctx_dim)
        # prefix: the sos token, with shape of (n_cls, 1, ctx_dim)
        # suffix: remaining tokens, with shape of (n_cls, *, ctx_dim)

        if label is not None:
            prefix = prefix[label]
            suffix = suffix[label]

        prompts = torch.cat(
            [
                prefix,  # (dim0, 1, dim)
                ctx,     # (dim0, n_ctx, dim)
                suffix,  # (dim0, *, dim)
            ],
            dim=1,
        )

        return prompts

    def forward(self, im_features):
        prefix = self.token_prefix
        suffix = self.token_suffix
        ctx = self.ctx                     # (n_ctx, ctx_dim)
        bias = self.meta_net(im_features)  # (batch, ctx_dim)
        bias = bias.unsqueeze(1)           # (batch, 1, ctx_dim)
        ctx = ctx.unsqueeze(0)             # (1, n_ctx, ctx_dim)
        ctx_shifted = ctx + bias           # (batch, n_ctx, ctx_dim)
        
        # Use instance-conditioned context tokens for all classes
        prompts = []
        for ctx_shifted_i in ctx_shifted:
            ctx_i = ctx_shifted_i.unsqueeze(0).expand(self.n_cls, -1, -1)
            pts_i = self.construct_prompts(ctx_i, prefix, suffix)  # (n_cls, n_tkn, ctx_dim)
            prompts.append(pts_i)
        prompts = torch.stack(prompts)
        
        return prompts


class Trigger(nn.Module):
    def __init__(self, cfg, dtype, device="cuda"):
        super().__init__()
        self.mean_as_tensor = torch.as_tensor(cfg.INPUT.PIXEL_MEAN, dtype=dtype, device=device).view(-1, 1, 1)
        self.std_as_tensor = torch.as_tensor(cfg.INPUT.PIXEL_STD, dtype=dtype, device=device).view(-1, 1, 1)
        self.lower_bound = (torch.zeros([1, 3, cfg.INPUT.SIZE[0], cfg.INPUT.SIZE[1]], device=device)
                            - self.mean_as_tensor) / self.std_as_tensor
        self.upper_bound = (torch.ones([1, 3, cfg.INPUT.SIZE[0], cfg.INPUT.SIZE[1]], device=device)
                            - self.mean_as_tensor) / self.std_as_tensor
        self.eps = cfg.BACKDOOR.EPS / 255.0
        self.trigger = nn.Parameter(
            (torch.rand([1, 3, cfg.INPUT.SIZE[0], cfg.INPUT.SIZE[1]], device=device) - 0.5) * 2 * self.eps / self.std_as_tensor, requires_grad=True)

        self.target = cfg.BACKDOOR.TARGET
        self.target_name = None

    def forward(self, image):
        # print(image)
        # print(self.trigger)
        # import numpy as np
        # import cv2
        # bd_images = torch.min(torch.max(image + self.trigger, self.lower_bound), self.upper_bound)
        # img = (bd_images * self.std_as_tensor + self.mean_as_tensor)[0]
        # mat = np.uint8(img.cpu().numpy().transpose(1, 2, 0) * 255)
        # mat = cv2.cvtColor(mat, cv2.COLOR_BGR2RGB)
        # cv2.imwrite("vis_images/0.png", mat)
        # exit()
        return torch.min(torch.max(image + self.trigger, self.lower_bound), self.upper_bound)

    def clamp(self):
        self.trigger.data = torch.min(torch.max(self.trigger.detach(), - self.eps / self.std_as_tensor),
                                 self.eps / self.std_as_tensor).data
        self.trigger.data = torch.min(torch.max(self.trigger.detach(), self.lower_bound), self.upper_bound).data

    def set_target_name(self, name):
        self.target_name = name

        return self.target_name

class CustomCLIP(nn.Module):
    def __init__(self, cfg, classnames, clip_model):
        super().__init__()
        self.prompt_learner = PromptLearner(cfg, classnames, clip_model)
        self.tokenized_prompts = self.prompt_learner.tokenized_prompts
        self.image_encoder = clip_model.visual
        self.text_encoder = TextEncoder(clip_model)
        self.logit_scale = clip_model.logit_scale
        self.dtype = clip_model.dtype

        self.trigger = Trigger(cfg, self.dtype)
        self.trigger.set_target_name(self.prompt_learner.classnames[int(self.trigger.target)])

    def forward(self, image, label=None):
        tokenized_prompts = self.tokenized_prompts
        logit_scale = self.logit_scale.exp()

        image_features = self.image_encoder(image.type(self.dtype))
        image_features = image_features / image_features.norm(dim=-1, keepdim=True)

        prompts = self.prompt_learner(image_features)
        
        logits = []
        for pts_i, imf_i in zip(prompts, image_features):
            text_features = self.text_encoder(pts_i, tokenized_prompts)
            text_features = text_features / text_features.norm(dim=-1, keepdim=True)
            l_i = logit_scale * imf_i @ text_features.t()
            logits.append(l_i)
        logits = torch.stack(logits)
        
        if self.prompt_learner.training:
            return F.cross_entropy(logits, label)
        
        return logits


@TRAINER_REGISTRY.register()
class BadClip(TrainerX):
    def check_cfg(self, cfg):
        assert cfg.TRAINER.COCOOP.PREC in ["fp16", "fp32", "amp"]

    def build_model(self):
        cfg = self.cfg
        classnames = self.dm.dataset.classnames

        print(f"Loading CLIP (backbone: {cfg.MODEL.BACKBONE.NAME})")
        clip_model = load_clip_to_cpu(cfg)
        
        if cfg.TRAINER.COCOOP.PREC == "fp32" or cfg.TRAINER.COCOOP.PREC == "amp":
            # CLIP's default precision is fp16
            clip_model.float()

        print("Building custom CLIP")
        self.model = CustomCLIP(cfg, classnames, clip_model)

        print("Turning off gradients in both the image and the text encoder")

        for name, param in self.model.named_parameters():
            if not ("prompt_learner" in name or "trigger" in name):
                param.requires_grad_(False)

        
        # Double check
        enabled = set()
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                enabled.add(name)
        print(f"Parameters to be updated: {enabled}")

        if cfg.MODEL.INIT_WEIGHTS:
            load_pretrained_weights(self.model.prompt_learner, cfg.MODEL.INIT_WEIGHTS)

        self.model.to(self.device)
        # NOTE: only give prompt_learner to the optimizer
        self.optim = build_optimizer(self.model.prompt_learner, cfg.OPTIM)
        self.sched = build_lr_scheduler(self.optim, cfg.OPTIM)
        self.register_model("prompt_learner", self.model.prompt_learner, self.optim, self.sched)


        self.trigger_optim = build_optimizer(self.model.trigger, cfg.OPTIM)
        self.trigger_sched = build_lr_scheduler(self.trigger_optim, cfg.OPTIM)
        self.register_model("trigger", self.model.trigger, self.trigger_optim, self.trigger_sched)

        self.scaler = GradScaler() if cfg.TRAINER.COCOOP.PREC == "amp" else None

        # Note that multi-gpu training could be slow because CLIP's size is
        # big, which slows down the copy operation in DataParallel
        device_count = torch.cuda.device_count()
        if device_count > 1:
            print(f"Multiple GPUs detected (n_gpus={device_count}), use all of them!")
            self.model = nn.DataParallel(self.model)

        self.test_on_new = False
        self.clip_model = clip_model

    def forward_backward_init_trigger(self, batch):
        image, label = self.parse_batch_train(batch)
        image = torch.cat((image, self.model.trigger(image.clone().detach())), dim=0)
        label = torch.cat((label, torch.zeros_like(label) + self.model.trigger.target), dim=0)

        model = self.model

        loss = model(image, label)
        loss.backward()
        model.trigger.trigger.data = model.trigger.trigger.data - 0.1 * self.model.trigger.trigger.grad.data
        model.trigger.clamp()
        model.trigger.zero_grad()
        model.prompt_learner.zero_grad()

        loss_summary = {"loss_init_trigger": loss.item()}

        return loss_summary

    def run_epoch_init_trigger(self):
        self.set_model_mode("train")
        losses = MetricMeter()
        batch_time = AverageMeter()
        data_time = AverageMeter()
        self.num_batches = len(self.train_loader_x)

        end = time.time()
        for self.batch_idx, batch in enumerate(self.train_loader_x):
            data_time.update(time.time() - end)
            loss_summary = self.forward_backward_init_trigger(batch)
            batch_time.update(time.time() - end)
            losses.update(loss_summary)

            meet_freq = (self.batch_idx + 1) % self.cfg.TRAIN.PRINT_FREQ == 0
            only_few_batches = self.num_batches < self.cfg.TRAIN.PRINT_FREQ
            if meet_freq or only_few_batches:
                nb_remain = 0
                nb_remain += self.num_batches - self.batch_idx - 1
                nb_remain += (
                    self.cfg.BACKDOOR.INIT.EPOCH - self.init_epoch - 1
                ) * self.num_batches
                eta_seconds = batch_time.avg * nb_remain
                eta = str(datetime.timedelta(seconds=int(eta_seconds)))

                info = []
                info += [f"epoch [{self.init_epoch + 1}/{self.cfg.BACKDOOR.INIT.EPOCH}]"]
                info += [f"batch [{self.batch_idx + 1}/{self.num_batches}]"]
                info += [f"time {batch_time.val:.3f} ({batch_time.avg:.3f})"]
                info += [f"data {data_time.val:.3f} ({data_time.avg:.3f})"]
                info += [f"{losses}"]
                info += [f"lr {self.cfg.BACKDOOR.INIT.LR:.4e}"]
                info += [f"eta {eta}"]
                print(" ".join(info))

            n_iter = self.init_epoch * self.num_batches + self.batch_idx
            for name, meter in losses.meters.items():
                self.write_scalar("init_trigger/" + name, meter.avg, n_iter)
            self.write_scalar("init_trigger/lr", self.cfg.BACKDOOR.INIT.LR, n_iter)

            end = time.time()

    def before_train(self):
        directory = self.cfg.OUTPUT_DIR
        if self.cfg.RESUME:
            directory = self.cfg.RESUME
        self.start_epoch = self.resume_model_if_exist(directory)

        # Initialize summary writer
        writer_dir = osp.join(self.output_dir, "tensorboard")
        mkdir_if_missing(writer_dir)
        self.init_writer(writer_dir)

        # Remember the starting time (for computing the elapsed time)
        self.time_start = time.time()

        if self.cfg.BACKDOOR.INIT.EXEC:
            print("Initialize trigger for {} epochs".format(self.cfg.BACKDOOR.INIT.EPOCH))
            for self.init_epoch in range(0, self.cfg.BACKDOOR.INIT.EPOCH):
                self.run_epoch_init_trigger()

    def after_train(self):
        print("Finish training")

        do_test = not self.cfg.TEST.NO_TEST
        if do_test:
            if self.cfg.TEST.FINAL_MODEL == "best_val":
                print("Deploy the model with the best val performance")
                self.load_model(self.output_dir)
            else:
                print("Deploy the last-epoch model")
            self.test()
            self.test_backdoor()

        # Show elapsed time
        elapsed = round(time.time() - self.time_start)
        elapsed = str(datetime.timedelta(seconds=elapsed))
        print(f"Elapsed: {elapsed}")

        # Close writer
        self.close_writer()

    def forward_backward(self, batch):
        image, label = self.parse_batch_train(batch)
        image = torch.cat((image, self.model.trigger(image.clone().detach())), dim=0)
        label = torch.cat((label, torch.zeros_like(label)+self.model.trigger.target), dim=0)

        model = self.model
        optim = self.optim
        trigger_optim = self.trigger_optim
        scaler = self.scaler
        
        prec = self.cfg.TRAINER.COCOOP.PREC
        if prec == "amp":
            with autocast():
                loss = model(image, label)
            optim.zero_grad()
            trigger_optim.zero_grad()
            scaler.scale(loss).backward()
            scaler.step(optim)
            scaler.step(trigger_optim)
            model.trigger.clamp()
            scaler.update()
        else:
            loss = model(image, label)
            optim.zero_grad()
            trigger_optim.zero_grad()
            loss.backward()
            optim.step()
            trigger_optim.step()
            model.trigger.clamp()

        loss_summary = {"loss": loss.item()}

        if (self.batch_idx + 1) == self.num_batches:
            self.update_lr()

        return loss_summary

    def parse_batch_train(self, batch):
        input = batch["img"]
        label = batch["label"]
        input = input.to(self.device)
        label = label.to(self.device)
        return input, label

    def load_model(self, directory, epoch=None):
        if not directory:
            print("Note that load_model() is skipped as no pretrained model is given")
            return

        names = self.get_model_names()

        # Load trigger first
        if "trigger" in names:
            names.remove("trigger")
            names.insert(0, "trigger")

        # By default, the best model is loaded
        model_file = "model-best.pth.tar"

        if epoch is not None:
            model_file = "model.pth.tar-" + str(epoch)

        for name in names:
            model_path = osp.join(directory, name, model_file)

            if not osp.exists(model_path):
                raise FileNotFoundError('Model not found at "{}"'.format(model_path))

            checkpoint = load_checkpoint(model_path)
            state_dict = checkpoint["state_dict"]
            epoch = checkpoint["epoch"]

            # Ignore fixed token vectors
            if "token_prefix" in state_dict:
                del state_dict["token_prefix"]

            if "token_suffix" in state_dict:
                del state_dict["token_suffix"]

            # Ignore classnames
            if "classnames" in state_dict:
                del state_dict["classnames"]

            print("Loading weights to {} " 'from "{}" (epoch = {})'.format(name, model_path, epoch))
            # set strict=False
            self._models[name].load_state_dict(state_dict, strict=False)

            if name == "trigger" and "target_name" in checkpoint.keys():
                self._models[name].set_target_name(checkpoint["target_name"])

            # Rebuild PromptLearner when testing on the new classes or new datasets
            if name == "trigger" and self.model.trigger.target_name != self.model.prompt_learner.classnames[self.model.trigger.target]:
                if self.model.trigger.target_name in self.model.prompt_learner.classnames:
                    self.model.trigger.target = self.model.prompt_learner.classnames.index(self.model.trigger.target_name)
                else:
                    print("Rebuilding PromptLearner by adding class: {}".format(self.model.trigger.target_name))
                    self.test_on_new = True
                    new_classnames = self.model.prompt_learner.classnames
                    new_classnames.insert(0, self.model.trigger.target_name)
                    self.model.trigger.target = 0
                    self.model.prompt_learner = PromptLearner(self.cfg, new_classnames, self.clip_model).to(self.device)
                    self.optim = build_optimizer(self.model.prompt_learner, self.cfg.OPTIM)
                    self.sched = build_lr_scheduler(self.optim, self.cfg.OPTIM)
                    self.overwrite_model("prompt_learner", self.model.prompt_learner, self.optim, self.sched)
                    self.model.tokenized_prompts = self.model.prompt_learner.tokenized_prompts

    @torch.no_grad()
    def test(self, split=None):
        """A generic testing pipeline."""
        self.set_model_mode("eval")
        self.evaluator.reset()

        if split is None:
            split = self.cfg.TEST.SPLIT

        if split == "val" and self.val_loader is not None:
            data_loader = self.val_loader
        else:
            split = "test"  # in case val_loader is None
            data_loader = self.test_loader

        print(f"Evaluate on the *{split}* set")

        for batch_idx, batch in enumerate(tqdm(data_loader)):
            input, label = self.parse_batch_test(batch)
            output = self.model_inference(input)
            # The first class is assigned as the target class when testing on the new classes or new datasets
            if self.test_on_new:
                label.data = label.data + 1
            self.evaluator.process(output, label)

        results = self.evaluator.evaluate()

        for k, v in results.items():
            tag = f"{split}/{k}"
            self.write_scalar(tag, v, self.epoch)

        return list(results.values())[0]

    @torch.no_grad()
    def test_backdoor(self, split=None):
        """A generic testing pipeline."""
        self.set_model_mode("eval")
        self.evaluator.reset()

        if split is None:
            split = self.cfg.TEST.SPLIT

        if split == "val" and self.val_loader is not None:
            data_loader = self.val_loader
        else:
            split = "test"  # in case val_loader is None
            data_loader = self.test_loader

        print(f"Evaluate on the backdoored *{split}* set")

        for batch_idx, batch in enumerate(tqdm(data_loader)):
            input, label = self.parse_batch_test(batch)
            label = torch.zeros_like(label) + self.model.trigger.target
            output = self.model_inference(self.model.trigger(input))
            self.evaluator.process(output, label)

        results = self.evaluator.evaluate()

        for k, v in results.items():
            tag = f"{split}/{k}"
            self.write_scalar(tag, v, self.epoch)

        return list(results.values())[0]

    def save_model(
        self, epoch, directory, is_best=False, val_result=None, model_name=""
    ):
        names = self.get_model_names()

        for name in names:
            model_dict = self._models[name].state_dict()

            optim_dict = None
            if self._optims[name] is not None:
                optim_dict = self._optims[name].state_dict()

            sched_dict = None
            if self._scheds[name] is not None:
                sched_dict = self._scheds[name].state_dict()

            if name == "trigger":
                save_checkpoint(
                    {
                        "state_dict": model_dict,
                        "epoch": epoch + 1,
                        "optimizer": optim_dict,
                        "scheduler": sched_dict,
                        "val_result": val_result,
                        "target_name": self._models[name].target_name
                    },
                    osp.join(directory, name),
                    is_best=is_best,
                    model_name=model_name,
                )
            else:
                save_checkpoint(
                    {
                        "state_dict": model_dict,
                        "epoch": epoch + 1,
                        "optimizer": optim_dict,
                        "scheduler": sched_dict,
                        "val_result": val_result
                    },
                    osp.join(directory, name),
                    is_best=is_best,
                    model_name=model_name,
                )

    def overwrite_model(self, name="model", model=None, optim=None, sched=None):
        if self.__dict__.get("_models") is None:
            raise AttributeError(
                "Cannot assign model before super().__init__() call"
            )

        if self.__dict__.get("_optims") is None:
            raise AttributeError(
                "Cannot assign optim before super().__init__() call"
            )

        if self.__dict__.get("_scheds") is None:
            raise AttributeError(
                "Cannot assign sched before super().__init__() call"
            )

        self._models[name] = model
        self._optims[name] = optim
        self._scheds[name] = sched