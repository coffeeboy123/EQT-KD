import data_v1
import data_v2
from loss import make_loss
from model import make_model_student, make_model_teacher
from optim import make_optimizer, make_scheduler

# import engine_v1
# import engine_v2
import engine_v3
import os.path as osp
from option import args
import utils.utility as utility
from utils.model_complexity import compute_model_complexity
from torch.utils.collect_env import get_pretty_env_info
import yaml
import torch


if __name__ == '__main__':
    if args.config != "":
        with open(args.config, "r") as f:
            config = yaml.full_load(f)
        for op in config:
            setattr(args, op, config[op])
    torch.backends.cudnn.benchmark = True

    # loader = data.Data(args)
    ckpt = utility.checkpoint(args)
    loader = data_v2.ImageDataManager(args)
    model_student = make_model_student(args, ckpt)

    ckpt.write_log(
        "[INFO] Model parameters: {com[0]} flops: {com[1]}".format(
            com=compute_model_complexity(model_student, (1, 3, args.height, args.width))
        )
    )

    model_student.qconfig = torch.quantization.get_default_qat_qconfig("qnnpack")
    model_student = torch.quantization.prepare_qat(model_student, inplace=True)

    model_teacher = make_model_teacher(args, ckpt)
    optimzer = make_optimizer(args, model_student)
    loss = make_loss(args, ckpt) if not args.test_only else None

    if args.teacher_pretrain != "":
        ckpt.load_pretrained_weights(model_teacher, args.teacher_pretrain)


    start = -1
    if args.load != "":
        start, model, optimizer = ckpt.resume_from_checkpoint(
            osp.join(ckpt.dir, "model-latest.pth"), model_student, optimzer
        )
        start = start - 1

    scheduler = make_scheduler(args, optimzer, start)

    # print('[INFO] System infomation: \n {}'.format(get_pretty_env_info()))


    engine = engine_v3.Engine(args, model_student, model_teacher, optimzer, scheduler, loss, loader, ckpt)
    # engine = engine.Engine(args, model, loss, loader, ckpt)

    n = start + 1
    while not engine.terminate():
        n += 1
        engine.train()
        engine.validation()

        # 🔽 100 에폭 이후부터 매 에폭마다 테스트
        if n >= 1:
            
            if args.test_every != 0 and n % args.test_every == 0:
                engine.test()
            elif n == args.epochs:
                engine.test()
    engine.close()
    # keep the final QAT (fake-quant, float) weights for tools/eval_fp32_fq_int8.py
    torch.save(model_student.state_dict(), osp.join(ckpt.dir, "qat_final_state.pth"))
