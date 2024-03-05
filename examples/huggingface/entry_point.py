from transformers import (
    get_linear_schedule_with_warmup,
    AutoModelForCausalLM,
    Trainer,
)
import torch
import torch.optim as optim

model_id = "roberta-base"


def deepview_model_provider():
    return AutoModelForCausalLM.from_pretrained(model_id, is_decoder=True).cuda()


def deepview_input_provider(batch_size=2):
    vocab_size = 30522
    src_seq_len = 512
    tgt_seq_len = 512 

    device = torch.device("cuda")

    source = torch.randint(
        low=0,
        high=vocab_size,
        size=(batch_size, src_seq_len),
        dtype=torch.int64,
        device=device,
    )
    target = torch.randint(
        low=0,
        high=vocab_size,
        size=(batch_size, tgt_seq_len),
        dtype=torch.int64,
        device=device,
    )
    return (source, target)


def deepview_iteration_provider(model):
    model.parameters()
    optimizer = optim.AdamW(
        params=model.parameters(),
        betas=(0.9, 0.999),
        eps=1e-6,
        weight_decay=0.01,
        lr=1e-4,
    )
    scheduler = get_linear_schedule_with_warmup(optimizer, 10000, 500000)
    trainer = Trainer(model=model, optimizers=(optimizer, scheduler))

    def iteration(source, label):
        trainer.training_step(model, {"input_ids": source, "labels": label})

    return iteration