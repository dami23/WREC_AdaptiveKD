import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import pdb

def kl_loss(student_logit, teacher_logit, temperature, adaptive_temperature):
    if not adaptive_temperature:
        distill_temperature = temperature
    else:
        MLP_module = nn.Sequential(nn.Linear(teacher_logit.size(0), 1),
                                   nn.Sigmoid()).cuda()
        sigmoid = nn.Sigmoid()

        teacher_entropy = - torch.sum(teacher_logit * torch.log(teacher_logit), dim=1)
        distill_temperature = sigmoid(MLP_module(teacher_entropy))
    
    student_out = F.log_softmax(student_logit / temperature, dim=-1)
    teacher_out = F.softmax(teacher_logit / distill_temperature, dim=-1)
    kl_loss = F.kl_div(student_out, teacher_out, reduction="none").sum(-1) * distill_temperature * temperature
    
    return kl_loss

def dynamic_knowledge_adaptive(student_logits, teacher_logits_target, teacher_logits_inter, temperature, adaptive_temperature):
    if not adaptive_temperature:
        target_temperature = inter_temperature = temperature
    else:
        MLP_module = nn.Sequential(nn.Linear(teacher_logits_target.size(0), 1),
                                   nn.Sigmoid()).cuda()
        sigmoid = nn.Sigmoid()

        target_entropy = - torch.sum(teacher_logits_target * torch.log(teacher_logits_target), dim=1)
        target_temperature = sigmoid(MLP_module(target_entropy))

        inter_entropy = - torch.sum(teacher_logits_inter * torch.log(teacher_logits_inter), dim=1)
        inter_temperature = sigmoid(MLP_module(inter_entropy))

    input = F.log_softmax(student_logits / temperature, dim=-1)
    target = F.softmax(teacher_logits_target / target_temperature, dim=-1)
    inter = F.softmax(teacher_logits_inter / inter_temperature, dim=-1)

    probs = F.softmax(student_logits, dim=-1)  # bsz, num_labels
    entropy = - torch.sum(probs * torch.log(probs), dim=1)
    avg_prob = 1 / student_logits.size(1) * torch.ones((1, student_logits.size(1)))
    confidence = - entropy / torch.sum(avg_prob * torch.log(avg_prob))

    target_loss = F.kl_div(input, target, reduction="none").sum(-1) * temperature * target_temperature
    inter_loss = F.kl_div(input, inter, reduction="none").sum(-1) * temperature * inter_temperature

    weighted_loss = torch.mean(confidence * target_loss, dim = 0) + torch.mean((1 - confidence) * inter_loss, dim = 0)

    return weighted_loss

def dynamic_knowledge_fixed_temp(student_logits, teacher_logits_target, teacher_logits_inter, temperature, distill_temperature):      
    input = F.log_softmax(student_logits / temperature, dim=-1)
    target = F.softmax(teacher_logits_target / distill_temperature, dim=-1)
    inter = F.softmax(teacher_logits_inter / distill_temperature, dim=-1)
    
    probs = F.softmax(student_logits, dim=-1)  # bsz, num_labels
    entropy = - torch.sum(probs * torch.log(probs), dim=1)
    avg_prob = 1 / student_logits.size(1) * torch.ones((1, student_logits.size(1)))
    confidence = - entropy / torch.sum(avg_prob * torch.log(avg_prob))

    target_loss = F.kl_div(input, target, reduction="none").sum(-1) * temperature * distill_temperature
    inter_loss = F.kl_div(input, inter, reduction="none").sum(-1) * temperature * distill_temperature

    weighted_loss = torch.mean(confidence * target_loss, dim = 0) + torch.mean((1 - confidence) * inter_loss, dim = 0)

    return weighted_loss