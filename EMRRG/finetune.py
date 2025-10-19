import os
import sys
from typing import List

import fire
import torch
import transformers
from datasets import load_dataset
from typing import List, Optional, Union


sys.path.append(os.path.join(os.getcwd(), "mamba_peft/src/"))
from mamba_peft.src.peft import (  # noqa: E402
    LoraConfig,
    BottleneckConfig,
    PrefixTuningConfig,
    MambaPEFTConfig,
    get_peft_model,
    get_peft_model_state_dict,
    prepare_model_for_int8_training,
    set_peft_model_state_dict,
)
from transformers import AutoModelForCausalLM, AutoTokenizer, LlamaTokenizer, AutoModel  # noqa: F402
from mamba_ssm.models.mixer_seq_simple import MambaLMHeadModel
from transformers import MambaForCausalLM


# From https://github.com/redotvideo/mamba-chat/blob/main/trainer/mamba_trainer.py
class MambaTrainer(transformers.Trainer):
    def compute_loss(self, model, inputs, return_outputs=False):
        input_ids = inputs.pop("input_ids")
        lm_logits = model(input_ids).logits

        labels = input_ids.to(lm_logits.device)
        shift_logits = lm_logits[:, :-1, :].contiguous()
        labels = labels[:, 1:].contiguous()

        loss_fct = torch.nn.CrossEntropyLoss()
        lm_loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), labels.view(-1))

        return lm_loss

    def save_model(self, output_dir, _internal_call):
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
            
        torch.save(self.model.state_dict(), f"{output_dir}/pytorch_model.bin")
        self.tokenizer.save_pretrained(output_dir)


def print_trainable_parameters(model):
    total_params = sum(p.numel() for p in model.parameters())
    total_trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total parameters: {total_params}")
    print(f"Total trainable parameters: {total_trainable_params}")
    print(f"Percentage of trainable parameters: {total_trainable_params / total_params * 100:.2f}%")


def train(
        # model/data params
        base_model: str = "state-spaces/mamba-130m-hf", 
        data_path: str = "./commonsense_170k.json",
        output_dir: str = "./output",
        adapter_name: str = "lora",
        load_8bit : bool = False,
        # training hyperparams
        batch_size: int = 128,
        micro_batch_size: int = 4,
        num_epochs: int = 3,
        learning_rate: float = 3e-4,
        weight_decay: float = 0.0,
        cutoff_len: int = 256,
        val_set_size: int = 2000,
        use_gradient_checkpointing: bool = False,
        eval_step: int = 200,
        save_step: int = 200,
        # lora hyperparams
        lora_r: int = 8,
        lora_alpha: int = 16,
        lora_dropout: float = 0.05,
        lora_target_modules: List[str] = None,
        # bottleneck adapter hyperparams
        bottleneck_size: int = 256,
        non_linearity: str = "tanh",
        adapter_dropout: float = 0.0,
        use_parallel_adapter: bool = False,
        use_adapterp: bool = False,
        target_modules: List[str] = None,
        scaling: Union[float, str] = 1.0,
        # prefix tuning hyperparams
        num_virtual_tokens: int = 30,
        # Additional scan hyperparams
        num_additional_dim: int = 6, 
        # Lora X
        dim_X=64,
        # Affix
        num_affix_virtual_tokens=3,
        # llm hyperparams
        train_on_inputs: bool = True,  # if False, masks out inputs in loss
        group_by_length: bool = False,  # faster, but produces an odd training loss curve
        # wandb params
        wandb_project: str = "",
        wandb_run_name: str = "",
        wandb_watch: str = "",  # options: false | gradients | all
        wandb_log_model: str = "",  # options: false | true
        resume_from_checkpoint: str = None,  # either training checkpoint or final adapter
):
    print(
        f"Finetuning model with params:\n"
        f"base_model: {base_model}\n"
        f"data_path: {data_path}\n"
        f"output_dir: {output_dir}\n"
        f"batch_size: {batch_size}\n"
        f"micro_batch_size: {micro_batch_size}\n"
        f"num_epochs: {num_epochs}\n"
        f"learning_rate: {learning_rate}\n"
        f"cutoff_len: {cutoff_len}\n"
        f"val_set_size: {val_set_size}\n"
        f"use_gradient_checkpointing: {use_gradient_checkpointing}\n"
        f"lora_r: {lora_r}\n"
        f"lora_alpha: {lora_alpha}\n"
        f"lora_dropout: {lora_dropout}\n"
        f"lora_target_modules: {lora_target_modules}\n"
        f"bottleneck_size: {bottleneck_size}\n"
        f"non_linearity: {non_linearity}\n"
        f"adapter_dropout: {adapter_dropout}\n"
        f"use_parallel_adapter: {use_parallel_adapter}\n"
        f"use_adapterp: {use_adapterp}\n"
        f"train_on_inputs: {train_on_inputs}\n"
        f"scaling: {scaling}\n"
        f"adapter_name: {adapter_name}\n"
        f"target_modules: {target_modules}\n"
        f"group_by_length: {group_by_length}\n"
        f"wandb_project: {wandb_project}\n"
        f"wandb_run_name: {wandb_run_name}\n"
        f"wandb_watch: {wandb_watch}\n"
        f"wandb_log_model: {wandb_log_model}\n"
        f"resume_from_checkpoint: {resume_from_checkpoint}\n"
    )
    assert (
        base_model
    ), "Please specify a --base_model, e.g. --base_model='decapoda-research/llama-7b-hf'"
    gradient_accumulation_steps = batch_size // micro_batch_size

    device_map = "auto"
    world_size = int(os.environ.get("WORLD_SIZE", 1))
    ddp = world_size != 1
    if ddp:
        print("ddp", world_size)
        device_map = {"": int(os.environ.get("LOCAL_RANK") or 0)}
        gradient_accumulation_steps = gradient_accumulation_steps // world_size

    # Check if parameter passed or if set within environ
    use_wandb = len(wandb_project) > 0 or (
            "WANDB_PROJECT" in os.environ and len(os.environ["WANDB_PROJECT"]) > 0
    )
    # Only overwrite environ if wandb param passed
    if len(wandb_project) > 0:
        os.environ["WANDB_PROJECT"] = wandb_project
    if len(wandb_watch) > 0:
        os.environ["WANDB_WATCH"] = wandb_watch
    if len(wandb_log_model) > 0:
        os.environ["WANDB_LOG_MODEL"] = wandb_log_model

    is_mamba = base_model.startswith("state-spaces/mamba-")
    is_llama = base_model.startswith("meta-llama")
    is_pythia = base_model.startswith("EleutherAI/pythia-")
    is_hf_model = base_model.endswith("hf")

    if is_mamba:
        if load_8bit:
            assert NotImplementedError, "We did not test Mamba model with 8-bit loading"
        elif is_hf_model:
            model = MambaForCausalLM.from_pretrained(
                base_model,
                load_in_8bit=False,
                torch_dtype=torch.float16,
                device_map=device_map, #a{"": int(os.environ.get("LOCAL_RANK") or 0)},
                trust_remote_code=True,
            )
        else:
            assert NotImplementedError, "For now, we only support Mamba model with HF model. Not the original mamba-ssm model"
            # model = MambaLMHeadModel.from_pretrained(base_model, device='cuda', dtype=torch.float16)
    else:
        if load_8bit:
            model = AutoModelForCausalLM.from_pretrained(
                base_model,
                load_in_8bit=load_8bit,
                torch_dtype=torch.float16,
                device_map=device_map,
                trust_remote_code=True,
            )
        else:
            model = AutoModelForCausalLM.from_pretrained(
                base_model,
                load_in_8bit=False,
                torch_dtype=torch.float16,
                device_map="auto", #a{"": int(os.environ.get("LOCAL_RANK") or 0)},
                trust_remote_code=True,
            )

    if is_mamba:
        tokenizer = AutoTokenizer.from_pretrained("EleutherAI/gpt-neox-20b")
    else:
        tokenizer = AutoTokenizer.from_pretrained(base_model, trust_remote_code=True)

    tokenizer.pad_token_id = (
        0  # unk. we want this to be different from the eos token
    )
    tokenizer.padding_side = "left"  # Allow batched inference

    def tokenize(prompt, add_eos_token=True):
        # there's probably a way to do this with the tokenizer settings
        # but again, gotta move fast
        result = tokenizer(
            prompt,
            truncation=True,
            max_length=cutoff_len,
            padding=False,
            return_tensors=None,
        )
        if (
                result["input_ids"][-1] != tokenizer.eos_token_id
                and len(result["input_ids"]) < cutoff_len
                and add_eos_token
        ):
            result["input_ids"].append(tokenizer.eos_token_id)
            if "chatglm" not in base_model:
                result["attention_mask"].append(1)

        result["labels"] = result["input_ids"].copy()

        if "chatglm" in base_model:
            return {"input_ids": result["input_ids"], "labels": result["labels"]}
        else:
            return result

    def generate_and_tokenize_prompt(data_point):
        full_prompt = generate_prompt(data_point)
        tokenized_full_prompt = tokenize(full_prompt)
        if not train_on_inputs:
            user_prompt = generate_prompt({**data_point, "output": ""})
            tokenized_user_prompt = tokenize(user_prompt, add_eos_token=False)
            user_prompt_len = len(tokenized_user_prompt["input_ids"])

            tokenized_full_prompt["labels"] = [
                                                  -100
                                              ] * user_prompt_len + tokenized_full_prompt["labels"][
                                                                    user_prompt_len:
                                                                    ]  # could be sped up, probably
        return tokenized_full_prompt

    # model = prepare_model_for_int8_training(model, use_gradient_checkpointing=use_gradient_checkpointing)
    model.float()
    print(model)

    if is_llama:
        if adapter_name == "lora":
            config = LoraConfig(
                r=lora_r,
                lora_alpha=lora_alpha,
                target_modules=target_modules,
                lora_dropout=lora_dropout,
                bias="none",
                task_type="CAUSAL_LM",
            )
        elif adapter_name == "bottleneck":
            config = BottleneckConfig(
                bottleneck_size=bottleneck_size,
                non_linearity=non_linearity,
                adapter_dropout=adapter_dropout,
                use_parallel_adapter=use_parallel_adapter,
                use_adapterp=use_adapterp,
                target_modules=target_modules,
                scaling=scaling,
                bias="none",
                task_type="CAUSAL_LM",
            )
        elif adapter_name == "prefix-tuning":
            config = PrefixTuningConfig(
                num_virtual_tokens=num_virtual_tokens,
                task_type="CAUSAL_LM",
            )
    elif is_pythia:
        if adapter_name == "full":
            pass
        elif adapter_name == "lora":
            config = LoraConfig(
                r=lora_r,
                lora_alpha=lora_alpha,
                lora_dropout=lora_dropout,
                bias="none",
                task_type="CAUSAL_LM",
            )
        else:
            assert NotImplementedError
    elif is_mamba:
        if adapter_name == "full":
            pass
        elif adapter_name == "lora_sll":
            if "130m" in base_model:
                dim_scale=1
            elif "370m" in base_model:
                dim_scale=2
            elif "790m" in base_model:
                dim_scale=4
            elif "1.4b" in base_model:
                dim_scale=8
            elif "2.8b" in base_model:
                dim_scale=16
            else:
                raise NotImplementedError
            config = MambaPEFTConfig(
                lora_out_proj=True,
                dim = dim_scale*8,
                lora_in_proj=True,
                dim_in_proj = dim_scale*8,
                lora_x_proj=True,
                dim_x_proj = dim_scale*8,
                lora_patch_embed=True,
                dim_patch_embed = dim_scale*8,
                task_type="CAUSAL_LM",
            )
        elif adapter_name == "lora_X":
            config = MambaPEFTConfig(
                lora_X=True,
                dim_X = dim_X if dim_X >0 else 64,
                s_X = 0.1,
                task_type="CAUSAL_LM",
            )
        elif adapter_name == "lora_out_proj":
            config = MambaPEFTConfig(
                lora_out_proj=True,
                dim_X = dim_X if dim_X >0 else 64,
                s_X = 0.1,
                task_type="CAUSAL_LM",
            )
        elif adapter_name == "lora_in_proj":
            config = MambaPEFTConfig(
                lora_in_proj=True,
                dim_X = dim_X if dim_X >0 else 64,
                s_X = 0.1,
                task_type="CAUSAL_LM",
            )
        elif adapter_name == "lora_x_proj":
            config = MambaPEFTConfig(
                lora_x_proj=True,
                dim_X = dim_X if dim_X >0 else 64,
                s_X = 0.1,
                task_type="CAUSAL_LM",
            )
        elif adapter_name == "lora_patch_embed":
            config = MambaPEFTConfig(
                lora_patch_embed=True,
                dim_X = dim_X if dim_X >0 else 64,
                s_X = 0.1,
                task_type="CAUSAL_LM",
            )
        elif "AddiScan" in adapter_name:
            config = MambaPEFTConfig(
                additional_scan=True,
                scan_A_copy_from_last=True,
                scan_addition_num=num_additional_dim if num_additional_dim>0 else 6,
                scan_addition_pos='prefix',
                zero_init_x_proj=False,
                task_type="CAUSAL_LM",
            )
        elif adapter_name=="AffixTuning":
            config = MambaPEFTConfig(
                prefix_tuning=True, 
                prefix_projection=False,
                prefix_type="inner_single_prefix", 
                num_virtual_tokens=num_affix_virtual_tokens,
                task_type="CAUSAL_LM",
            )
        elif "HybridPEFT_optimized_for_vision" in adapter_name:
            config = MambaPEFTConfig(
                additional_scan=False,
                dim=8,
                dim_B=4,
                dim_B_f=4,
                dim_C=4,
                dim_C_f=4,
                dim_X=8,
                dim_Z=8,
                dim_d=4,
                dim_d_f=4,
                dim_dt=4,
                dim_in_proj=8,
                dim_patch_embed=8,
                dim_x_proj=4,
                dim_x_proj_f=4,
                learnable_A=True,
                learnable_A_v2=True,
                learnable_D=False,
                learnable_D_v2=True,
                learnable_cls_token=False,
                learnable_cls_token_v2=True,
                learnable_conv1d=False,
                learnable_conv1d_v2=True,
                learnable_pos_embed=False,
                learnable_pos_embed_v2=True,
                lora_B=False,
                lora_C=False,
                lora_X=True,
                lora_Z=False,
                lora_d=False,
                lora_dt=True,
                lora_in_proj=True,
                lora_out_proj=True,
                lora_patch_embed=False,
                lora_x_proj=False,
                num_virtual_tokens=1,
                prefix_projection=True,
                prefix_tuning=True,
                prefix_type='inner_single_prefix',
                prompt_num_tokens=1,
                prompt_projection=True,
                prompt_tuning=False,
                prompt_type='prefix',
                s=0.1,
                s_B=0.1,
                s_C=0.1,
                s_X=0.1,
                s_Z=0.1,
                s_d=0.1,
                s_dt=0.1,
                s_in_proj=0.1,
                s_patch_embed=1,
                s_x_proj=0.1,
                scan_A_constant=None,
                scan_A_copy_from_last=True,
                scan_addition_num=1,
                scan_addition_pos='prefix',
                task_type="CAUSAL_LM",
            )
        else:
            assert NotImplementedError

    # set adapter without fine-tuning setting
    if not adapter_name == "full":
        model = get_peft_model(model, config)
    if adapter_name == "prefix-tuning":
        model.to('cuda')
    # enable gradient for mamba full-fineturning
    if is_mamba and is_hf_model and adapter_name == "full":
        # model.gradient_checkpointing_enable()
        # model.enable_input_require_grads()
        for params in model.parameters():
            params.requires_grad = True

    if data_path.endswith(".json"):  # todo: support jsonl
        data = load_dataset("json", data_files=data_path)
    else:
        data = load_dataset(data_path)

    if resume_from_checkpoint:
        # Check the available weights and load them
        checkpoint_name = os.path.join(
            resume_from_checkpoint, "pytorch_model.bin"
        )  # Full checkpoint
        if not os.path.exists(checkpoint_name) and not adapter_name == "full":
            checkpoint_name = os.path.join(
                resume_from_checkpoint, "adapter_model.bin"
            )  # only LoRA model - LoRA config above has to fit
            resume_from_checkpoint = (
                False  # So the trainer won't try loading its state
            )
        # The two files above have a different name depending on how they were saved, but are actually the same.
        if os.path.exists(checkpoint_name):
            print(f"Restarting from {checkpoint_name}")
            adapters_weights = torch.load(checkpoint_name)
            model = set_peft_model_state_dict(model, adapters_weights)
        else:
            print(f"Checkpoint {checkpoint_name} not found")

    print_trainable_parameters(model)

    if val_set_size > 0:
        train_val = data["train"].train_test_split(
            test_size=val_set_size, shuffle=True, seed=42
        )
        train_data = (
            train_val["train"].shuffle().map(generate_and_tokenize_prompt)
        )
        val_data = (
            train_val["test"].shuffle().map(generate_and_tokenize_prompt)
        )
    else:
        train_data = data["train"].shuffle().map(generate_and_tokenize_prompt)
        val_data = None

    if not ddp and torch.cuda.device_count() > 1:
        # keeps Trainer from trying its own DataParallelism when more than 1 gpu is available
        # model.is_parallelizable = True
        # model.model_parallel = True
        print("data parallel")
    if "HybridPEFT" in adapter_name or adapter_name=="AddiScan" or adapter_name=="AffixTuning" or adapter_name=="lora_sll" or  adapter_name=="lora_X":
        if adapter_name == "HybridPEFT_optimized_for_vision":
            trainable = []
            for n, p in model.named_parameters():
                # if 'A_log_scan_addi' in n and p.requires_grad:
                #     trainable.append({'params': p, 'weight_decay': weight_decay*0.1504988900782524, 'lr': learning_rate*0.20217551118282776})
                # elif 'head' in n and p.requires_grad:
                #     trainable.append({'params': p, 'weight_decay': weight_decay*5.825029745040104, 'lr': learning_rate*0.17793908309488815})
                # elif 'learnable_A' in n and p.requires_grad:
                #     trainable.append({'params': p, 'weight_decay': weight_decay*0.10118844775396835, 'lr': learning_rate*0.6880616999773654})
                # elif 'learnable_D' in n and p.requires_grad:
                #     trainable.append({'params': p, 'weight_decay': weight_decay*21.181836680418158, 'lr': learning_rate*0.16246153559065213})
                # elif 'learnable_cls_token' in n and p.requires_grad:
                #     trainable.append({'params': p, 'weight_decay': weight_decay*10., 'lr': learning_rate*1.})
                # elif 'learnable_conv1d' in n and p.requires_grad:
                #     trainable.append({'params': p, 'weight_decay': weight_decay*10, 'lr': learning_rate*1.})
                # elif 'learnable_patch_embed' in n and p.requires_grad:
                #     trainable.append({'params': p, 'weight_decay': weight_decay*10, 'lr': learning_rate*1})
                # elif 'lora_B' in n and p.requires_grad:
                #     trainable.append({'params': p, 'weight_decay': weight_decay*0.36383629195705464, 'lr': learning_rate*0.48072034617180165})
                # elif 'lora_C' in n and p.requires_grad:
                #     trainable.append({'params': p, 'weight_decay': weight_decay*1, 'lr': learning_rate*1})
                # elif 'lora_X' in n and p.requires_grad:
                #     trainable.append({'params': p, 'weight_decay': weight_decay*1, 'lr': learning_rate*1})
                # elif 'lora_Z' in n and p.requires_grad:
                #     trainable.append({'params': p, 'weight_decay': weight_decay*1, 'lr': learning_rate*1})
                # elif 'lora_d' in n and p.requires_grad:
                #     trainable.append({'params': p, 'weight_decay': weight_decay*1, 'lr': learning_rate*1})
                # elif 'lora_dt' in n and p.requires_grad:
                #     trainable.append({'params': p, 'weight_decay': weight_decay*0.029564762306324075, 'lr': learning_rate*0.1786857312741866})
                # elif 'lora_in_proj' in n and p.requires_grad:
                #     trainable.append({'params': p, 'weight_decay': weight_decay*1, 'lr': learning_rate*1})
                # elif 'lora_out_proj' in n and p.requires_grad:
                #     trainable.append({'params': p, 'weight_decay': weight_decay*0.03581685812857395, 'lr': learning_rate*1.1486185681597576})
                # elif 'lora_patch_embed' in n and p.requires_grad:
                #     trainable.append({'params': p, 'weight_decay': weight_decay*1, 'lr': learning_rate*1})
                # elif 'lora_x_proj' in n and p.requires_grad:
                #     trainable.append({'params': p, 'weight_decay': weight_decay*1, 'lr': learning_rate*1})
                # elif 'prefix_encoder' in n and p.requires_grad:
                #     trainable.append({'params': p, 'weight_decay': weight_decay*0.018549353285550397, 'lr': learning_rate*0.6536613594299218})
                # elif 'prompt_encoder' in n and p.requires_grad:
                #     trainable.append({'params': p, 'weight_decay': weight_decay*0.08416389699257418, 'lr': learning_rate*3.364257888150925})
                # elif 'x_proj_scan_addi' in n and p.requires_grad:
                #     trainable.append({'params': p, 'weight_decay': weight_decay*9.781887266526315, 'lr': learning_rate*0.2757764085007619})
                # elif p.requires_grad:
                #     trainable.append({'params': p})
                if p.requires_grad:
                    trainable.append({'params': p})
        elif adapter_name=="AddiScan":
            trainable = []
            for n, p in model.named_parameters():
                if 'A_log_scan_addi' in n and p.requires_grad:
                    print(n)
                    trainable.append({'params': p, 'weight_decay': 0, 'lr': learning_rate})
                elif p.requires_grad:
                    print(n)
                    trainable.append({'params': p})
        elif adapter_name=="AffixTuning" or adapter_name=="lora_sll" or adapter_name=="lora_X":
            trainable = []
            for n, p in model.named_parameters():
                if p.requires_grad:
                    print(n)
                    trainable.append({'params': p})
        

        from transformers import get_linear_schedule_with_warmup
        optimizer = torch.optim.AdamW(trainable, lr=learning_rate, weight_decay=weight_decay)
        scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=100, num_training_steps=(num_epochs*len(train_data))//batch_size + 1)
        trainer = transformers.Trainer(
            model=model,
            train_dataset=train_data,
            eval_dataset=val_data,
            args=transformers.TrainingArguments(
                per_device_train_batch_size=micro_batch_size,
                gradient_accumulation_steps=gradient_accumulation_steps,
                # warmup_steps=100,
                num_train_epochs=num_epochs,
                # learning_rate=learning_rate,
                # weight_decay=weight_decay,
                fp16=False,
                logging_steps=10,
                evaluation_strategy="steps" if val_set_size > 0 else "no",
                save_strategy="steps",
                eval_steps=eval_step if val_set_size > 0 else None,
                save_steps=save_step,
                output_dir=output_dir,
                save_total_limit=1,
                load_best_model_at_end=True if val_set_size > 0 else False,
                ddp_find_unused_parameters=False if ddp else None,
                group_by_length=group_by_length,
                report_to="wandb" if use_wandb else None,
                run_name=wandb_run_name if use_wandb else None,
            ),
            data_collator=transformers.DataCollatorForSeq2Seq(
                tokenizer, pad_to_multiple_of=8, return_tensors="pt", padding=True
            ),
             optimizers=(optimizer, scheduler)
        )
    else:
        trainer = transformers.Trainer(
        model=model,
        train_dataset=train_data,
        eval_dataset=val_data,
        args=transformers.TrainingArguments(
            per_device_train_batch_size=micro_batch_size,
            gradient_accumulation_steps=gradient_accumulation_steps,
            warmup_steps=100,
            num_train_epochs=num_epochs,
            learning_rate=learning_rate,
            weight_decay=weight_decay,
            fp16=False,
            logging_steps=10,
            optim="adamw_torch",
            evaluation_strategy="steps" if val_set_size > 0 else "no",
            save_strategy="steps",
            eval_steps=eval_step if val_set_size > 0 else None,
            save_steps=save_step,
            output_dir=output_dir,
            save_total_limit=1,
            load_best_model_at_end=True if val_set_size > 0 else False,
            ddp_find_unused_parameters=False if ddp else None,
            group_by_length=group_by_length,
            report_to="wandb" if use_wandb else None,
            run_name=wandb_run_name if use_wandb else None,
        ),
        data_collator=transformers.DataCollatorForSeq2Seq(
            tokenizer, pad_to_multiple_of=8, return_tensors="pt", padding=True
        ),
    )
    model.config.use_cache = False

    old_state_dict = model.state_dict

    if adapter_name != "full":
        model.state_dict = (
            lambda self, *_, **__: get_peft_model_state_dict(
                self, old_state_dict()
            )
        ).__get__(model, type(model))
    else:
        pass

    if torch.__version__ >= "2" and sys.platform != "win32":
        model = torch.compile(model)

    trainer.train(resume_from_checkpoint=resume_from_checkpoint)

    model.save_pretrained(output_dir)

    print(
        "\n If there's a warning about missing keys above, please disregard :)"
    )

    


def generate_prompt(data_point):
    # sorry about the formatting disaster gotta move fast
    if data_point["input"]:
        return f"""Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request. 

                ### Instruction:
                {data_point["instruction"]}
                
                ### Input:
                {data_point["input"]}
                
                ### Response:
                {data_point["output"]}""" # noqa: E501
    else:
        return f"""Below is an instruction that describes a task. Write a response that appropriately completes the request.  

                ### Instruction:
                {data_point["instruction"]}
                
                ### Response:
                {data_point["output"]}""" # noqa: E501


if __name__ == "__main__":
    fire.Fire(train)