# import time
# from collections import defaultdict
# import torch
# import torch.nn.functional as F

# from local import optimizer

# def validate(val_loader, utt2systemID_file, model, device, log_interval):
#     batch_time = optimizer.AverageMeter()
#     losses = optimizer.AverageMeter()
#     top1 = optimizer.AverageMeter()

#     # switch to evaluate mode
#     model.eval()
#     utt2scores = defaultdict(list)
#     with torch.no_grad():
#         end = time.time()
#         for i, (utt_list, input, target) in enumerate(val_loader):
#             input  = input.to(device, non_blocking=True)
#             target = target.to(device, non_blocking=True).view((-1,))

#             # compute output
#             output = model(input)

#             # loss 
#             loss = F.nll_loss(output, target)

#             # measure accuracy and record loss
#             acc1, = optimizer.accuracy(output, target, topk=(1, ))
#             losses.update(loss.item(), input.size(0))
#             top1.update(acc1[0], input.size(0))

#             # measure elapsed time
#             batch_time.update(time.time() - end)
#             end = time.time()

#             if i % log_interval == 0:
#                 print('Test: [{0}/{1}]\t'
#                       'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
#                       'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
#                       'Acc@1 {top1.val:.3f} ({top1.avg:.3f})'.format(
#                        i, len(val_loader), batch_time=batch_time, loss=losses,
#                        top1=top1))

#     print('===> Acc@1 {top1.avg:.3f}\n'.format(top1=top1))

#     return top1.avg

import time
from collections import defaultdict
import torch
import torch.nn.functional as F

from local import optimizer


def validate(val_loader, utt2systemID_file, model, device, log_interval, loss_func):
    batch_time = optimizer.AverageMeter()
    losses = optimizer.AverageMeter()
    top1 = optimizer.AverageMeter()
    lambda_ab = 0.1
    # switch to evaluate mode
    model.eval()
    utt2scores = defaultdict(list)
    with torch.no_grad():
        end = time.time()
        for i, (_,input, target) in enumerate(val_loader):
            # if i == 5:
            #     break
            input = input.to(device, non_blocking=True)
            target = target.to(device, non_blocking=True).view((-1,))
            output,ab_output,_ = model(input)
            # compute logits
            # loss
            loss_pb = loss_func(output, target)
            loss_ab = F.nll_loss(ab_output, target,torch.tensor(loss_func.weights).to(device))
            loss = loss_pb + loss_ab*lambda_ab
            logits = output[1]
            # measure accuracy and record loss
            acc1, = optimizer.accuracy(logits, target, topk=(1, ))
            losses.update(loss.item(), input.size(0))
            top1.update(acc1[0], input.size(0))

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            if i % log_interval == 0:
                print('Test: [{0}/{1}]\t'
                      'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                      'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                      'Acc@1 {top1.val:.3f} ({top1.avg:.3f})'.format(
                          i, len(val_loader), batch_time=batch_time, loss=losses,
                          top1=top1))

    print('===> Acc@1 {top1.avg:.3f}\n'.format(top1=top1))

    return top1.avg
