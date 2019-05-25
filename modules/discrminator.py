# -*- coding: utf-8 -*-

####################### 判别器网络 ############################
# 三个判别器，分别是：
# 1. 合成图像区域判别器
# 1. 合成图像判别器
# 1. 图像文本匹配判别器
##### reference


import torch
import torch.nn as nn
from torch.nn.utils.clip_grad import clip_grad_norm

from model import EncoderImage


def conv3x3(in_planes, out_planes, stride=1):
    "3x3 convolution with padding"
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)


def l1norm(X, dim, eps=1e-8):
    """L1-normalize columns of X
    """
    norm = torch.abs(X).sum(dim=dim, keepdim=True) + eps
    X = torch.div(X, norm)
    return X


def l2norm(X, dim, eps=1e-8):
    """L2-normalize columns of X
    """
    norm = torch.pow(X, 2).sum(dim=dim, keepdim=True).sqrt() + eps
    X = torch.div(X, norm)
    return X


def func_attention(query, context, arg, smooth, eps=1e-8):
    """
    query: (n_context, queryL, d)
    context: (n_context, sourceL, d)
    """
    batch_size_q, queryL = query.size(0), query.size(1)
    batch_size, sourceL = context.size(0), context.size(1)

    # Get attention
    # --> (batch, d, queryL)
    queryT = torch.transpose(query, 1, 2)

    # (batch, sourceL, d)(batch, d, queryL)
    # --> (batch, sourceL, queryL)
    attn = torch.bmm(context, queryT)
    if arg['raw_feature_norm'] == "softmax":
        # --> (batch*sourceL, queryL)
        attn = attn.view(batch_size * sourceL, queryL)
        attn = nn.Softmax()(attn)
        # --> (batch, sourceL, queryL)
        attn = attn.view(batch_size, sourceL, queryL)
    elif arg['raw_feature_norm'] == "l2norm":
        attn = l2norm(attn, 2)
    elif arg['raw_feature_norm'] == "clipped_l2norm":
        attn = nn.LeakyReLU(0.1)(attn)
        attn = l2norm(attn, 2)
    elif arg['raw_feature_norm'] == "l1norm":
        attn = l1norm_d(attn, 2)
    elif arg['raw_feature_norm'] == "clipped_l1norm":
        attn = nn.LeakyReLU(0.1)(attn)
        attn = l1norm_d(attn, 2)
    elif arg['raw_feature_norm'] == "clipped":
        attn = nn.LeakyReLU(0.1)(attn)
    elif opt.raw_feature_norm == "no_norm":
        pass
    else:
        raise ValueError("unknown first norm type:", arg['raw_feature_norm'])
    # --> (batch, queryL, sourceL)
    attn = torch.transpose(attn, 1, 2).contiguous()
    # --> (batch*queryL, sourceL)
    attn = attn.view(batch_size * queryL, sourceL)
    attn = nn.Softmax()(attn * smooth)
    # --> (batch, queryL, sourceL)
    attn = attn.view(batch_size, queryL, sourceL)
    # --> (batch, sourceL, queryL)
    attnT = torch.transpose(attn, 1, 2).contiguous()

    # --> (batch, d, sourceL)
    contextT = torch.transpose(context, 1, 2)
    # (batch x d x sourceL)(batch x sourceL x queryL)
    # --> (batch, d, queryL)
    weightedContext = torch.bmm(contextT, attnT)
    # --> (batch, queryL, d)
    weightedContext = torch.transpose(weightedContext, 1, 2)

    return weightedContext, attnT


def cosine_similarity(x1, x2, dim=1, eps=1e-8):
    """Returns cosine similarity between x1 and x2, computed along dim."""
    w12 = torch.sum(x1 * x2, dim)
    w1 = torch.norm(x1, 2, dim)
    w2 = torch.norm(x2, 2, dim)
    return (w12 / (w1 * w2).clamp(min=eps)).squeeze()


def xattn_score_t2i(images, captions, arg):
    """
    Images: (n_image, n_regions, d) matrix of images
    Captions: (n_caption, max_n_word, d) matrix of captions
    CapLens: (n_caption) array of caption lengths
    """
    similarities = []
    n_image = images.size(0)
    n_caption = captions.size(0)
    for i in range(n_caption):
        # Get the i-th text description
        cap_i = captions[i, :, :].unsqueeze(0).contiguous()
        # --> (n_image, r, d)
        cap_i_expand = cap_i.repeat(n_image, 1, 1)
        """
            word(query): (n_image, n_word, d)
            image(context): (n_image, n_regions, d)
            weiContext: (n_image, n_word, d)
            attn: (n_image, n_region, n_word)
        """
        weiContext, attn = func_attention(cap_i_expand, images, arg, smooth=arg['lambda_softmax'])
        cap_i_expand = cap_i_expand.contiguous()
        weiContext = weiContext.contiguous()
        # (n_image, n_word)
        row_sim = cosine_similarity(cap_i_expand, weiContext, dim=2)
        if arg['agg_func'] == 'LogSumExp':
            row_sim.mul_(arg['lambda_lse']).exp_()
            row_sim = row_sim.sum(dim=1, keepdim=True)
            row_sim = torch.log(row_sim) / arg['lambda_lse']
        elif arg['agg_func'] == 'Max':
            row_sim = row_sim.max(dim=1, keepdim=True)[0]
        elif arg['agg_func'] == 'Sum':
            row_sim = row_sim.sum(dim=1, keepdim=True)
        elif arg['agg_func'] == 'Mean':
            row_sim = row_sim.mean(dim=1, keepdim=True)
        else:
            raise ValueError("unknown aggfunc: {}".format(arg['agg_func']))
        similarities.append(row_sim)

    # (n_image, n_caption)
    similarities = torch.cat(similarities, 1)

    return similarities


def xattn_score_i2t(images, captions, opt):
    """
    Images: (batch_size, n_regions, d) matrix of images
    Captions: (batch_size, max_n_words, d) matrix of captions
    CapLens: (batch_size) array of caption lengths
    """
    similarities = []
    n_image = images.size(0)
    n_caption = captions.size(0)
    n_region = images.size(1)
    for i in range(n_caption):
        # Get the i-th text description
        cap_i = captions[i, :, :].unsqueeze(0).contiguous()
        # (n_image, r, d)
        cap_i_expand = cap_i.repeat(n_image, 1, 1)
        """
            word(query): (n_image, n_word, d)
            image(context): (n_image, n_region, d)
            weiContext: (n_image, n_region, d)
            attn: (n_image, n_word, n_region)
        """
        weiContext, attn = func_attention(images, cap_i_expand, opt, smooth=opt.lambda_softmax)
        # (n_image, n_region)
        row_sim = cosine_similarity(images, weiContext, dim=2)
        if opt.agg_func == 'LogSumExp':
            row_sim.mul_(opt.lambda_lse).exp_()
            row_sim = row_sim.sum(dim=1, keepdim=True)
            row_sim = torch.log(row_sim) / opt.lambda_lse
        elif opt.agg_func == 'Max':
            row_sim = row_sim.max(dim=1, keepdim=True)[0]
        elif opt.agg_func == 'Sum':
            row_sim = row_sim.sum(dim=1, keepdim=True)
        elif opt.agg_func == 'Mean':
            row_sim = row_sim.mean(dim=1, keepdim=True)
        else:
            raise ValueError("unknown aggfunc: {}".format(opt.agg_func))
        similarities.append(row_sim)

    # (n_image, n_caption)
    similarities = torch.cat(similarities, 1)
    return similarities


class ContrastiveLoss(nn.Module):
    """
    Compute contrastive loss
    """

    def __init__(self, arg, margin=0, max_violation=False):
        super(ContrastiveLoss, self).__init__()
        self.arg = arg
        self.margin = margin
        self.max_violation = max_violation

    def forward(self, im, s):
        # compute image-sentence score matrix
        if self.arg['cross_attn'] == 't2i':
            scores = xattn_score_t2i(im, s, self.arg)
        elif self.arg['cross_attn'] == 'i2t':
            scores = xattn_score_i2t(im, s, self.arg)
        else:
            raise ValueError("unknown first norm type:", self.arg['raw_feature_norm'])
        diagonal = scores.diag().view(im.size(0), 1)
        d1 = diagonal.expand_as(scores)
        d2 = diagonal.t().expand_as(scores)

        # compare every diagonal score to scores in its column
        # caption retrieval
        cost_s = (self.margin + scores - d1).clamp(min=0)
        # compare every diagonal score to scores in its row
        # image retrieval
        cost_im = (self.margin + scores - d2).clamp(min=0)

        # clear diagonals
        mask = torch.eye(scores.size(0)) > .5
        # I = Variable(mask)
        I = mask
        if torch.cuda.is_available():
            I = I.cuda()
        cost_s = cost_s.masked_fill_(I, 0)
        cost_im = cost_im.masked_fill_(I, 0)

        # keep the maximum violating negative for each query
        if self.max_violation:
            cost_s = cost_s.max(1)[0]
            cost_im = cost_im.max(0)[0]
        return cost_s.sum() + cost_im.sum()


class MatchDiscriminator(nn.Module):
    def __init__(self, arg, region_generator, region_discriminator):
        super(MatchDiscriminator, self).__init__()

        self.img_enc = EncoderImage(arg['img_dim'], arg['common_size'],
                                    precomp_enc_type=arg['precomp_enc_type'],
                                    no_imgnorm=arg['no_imgnorm'])

        self.txt_enc = nn.Linear(4 * 4 * arg['ndf'] * 8, arg['common_size']/2)

        self.region_generator = region_generator
        self.region_discriminator = region_discriminator

        if torch.cuda.is_available():
            self.img_enc.cuda()
            self.txt_enc.cuda()
            self.region_generator.cuda()
            self.region_discriminator.cuda()

        # Loss and Optimizer
        self.criterion = ContrastiveLoss(arg,
                                         margin=arg['margin'],
                                         max_violation=arg['max_violation'])
        params = list(self.region_generator.parameters())
        params += list(self.region_discriminator.parameters())
        params += list(self.txt_enc.parameters())
        params += list(self.img_enc.fc.parameters())

        self.params = params

        self.optimizer = torch.optim.Adam(params, lr=arg['match_learning_rate'])

        self.Eiters = 0
        self.r = arg['r']
        self.grad_clip = arg['grad_clip']

    def state_dict(self):
        state_dict = [self.img_enc.state_dict(), self.txt_enc.state_dict()]
        return state_dict

    def load_state_dict(self, state_dict):
        self.img_enc.load_state_dict(state_dict[0])
        self.txt_enc.load_state_dict(state_dict[1])

    def forward(self, images, captions, lengths):
        """Compute the image and caption embeddings
               """
        # Set mini-batch dataset
        if torch.cuda.is_available():
            images = images.cuda()
            captions = captions.cuda()

        # Forward
        img_emb = self.img_enc(images)

        fake_regions, penal, hidden, mu, embed = self.region_generator(captions, lengths)
        region_feature, fake_region_scores = self.region_discriminator(fake_regions)

        txt_emb = self.txt_enc(region_feature)
        txt_emb = txt_emb.view(-1, self.r, txt_emb.size(1))
        cap_emb = torch.cat([embed, txt_emb], 2)

        return img_emb, cap_emb

    def forward_loss(self, img_emb, cap_emb, **kwargs):
        """Compute the loss given pairs of image and caption embeddings
        """
        loss = self.criterion(img_emb, cap_emb)
        # print('Le', loss.data, img_emb.size(0))
        return loss

    def train_emb(self, images, captions, lengths, epoch):
        """One training step given images and captions.
        """
        self.Eiters += 1

        # compute the embeddings
        img_emb, cap_emb = self.forward(images, captions, lengths)

        # measure accuracy and record loss
        self.optimizer.zero_grad()
        loss = self.forward_loss(img_emb, cap_emb)

        # compute gradient and do SGD step
        loss.backward()
        if self.grad_clip > 0:
            clip_grad_norm(self.params, self.grad_clip)
        self.optimizer.step()

        print("Epoch: %d, iteration: %d, loss: %f " % (epoch, self.Eiters, loss.data))


class STAGE1_D(nn.Module):
    def __init__(self, arguments):
        super(STAGE1_D, self).__init__()
        self.image_feature_size = arguments["image_feature_size"]
        self.ndf = arguments['ndf']
        self.nef = arguments["condition_dimension"]
        self.sentence_embedding_size = arguments["embed_size"]
        self.project_size = arguments["project_size"]
        ndf, nef = self.ndf, self.nef

        self.bcondition = arguments["bcondition"]
        # input image size 3*64*64
        self.encode_image = nn.Sequential(
            nn.Conv2d(3, ndf, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf) x 32 x 32
            nn.Conv2d(ndf, ndf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 2),
            nn.LeakyReLU(0.2, inplace=True),
            # state size (ndf*2) x 16 x 16
            nn.Conv2d(ndf * 2, ndf * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 4),
            nn.LeakyReLU(0.2, inplace=True),
            # state size (ndf*4) x 8 x 8
            nn.Conv2d(ndf * 4, ndf * 8, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 8),
            # state size (ndf * 8) x 4 x 4)
            nn.LeakyReLU(0.2, inplace=True)
        )
        if self.bcondition:
            self.outlogits = nn.Sequential(
                conv3x3(ndf * 8 + nef, ndf * 8),
                nn.BatchNorm2d(ndf * 8),
                nn.LeakyReLU(0.2, inplace=True),
                nn.Conv2d(ndf * 8, 1, kernel_size=4, stride=4),
                nn.Sigmoid())
        else:
            self.outlogits = nn.Sequential(
                nn.Conv2d(ndf * 8, 1, kernel_size=4, stride=4),
                nn.Sigmoid())

    def forward(self, image, condition=None):
        image = image.view(-1, image.size(2), image.size(3), image.size(4))
        image_feature = self.encode_image(image)

        if self.bcondition:
            condition = condition.view(-1, self.nef, 1, 1)
            condition = condition.repeat(1, 1, 4, 4)
            # state size (ngf+nef) x 4 x 4
            code = torch.cat((image_feature, condition), 1)
        else:
            code = image_feature

        output = self.outlogits(code)
        return image_feature.view(-1, (self.ndf * 8) * 4 * 4), output.view(-1)


class STAGE2_D(nn.Module):
    def __init__(self, arg):
        super(STAGE2_D, self).__init__()
        self.df_dim = arg['ndf']
        self.ef_dim = arg["condition_dimension"]
        self.bcondition = arg["bcondition"]
        self.define_module()

    def define_module(self):
        ndf, nef = self.df_dim, self.ef_dim

        self.encode_image = nn.Sequential(
            nn.Conv2d(3, ndf, 4, 2, 1, bias=False),  # 128 * 128 * ndf
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(ndf, ndf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 2),
            nn.LeakyReLU(0.2, inplace=True),  # 64 * 64 * ndf * 2
            nn.Conv2d(ndf * 2, ndf * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 4),
            nn.LeakyReLU(0.2, inplace=True),  # 32 * 32 * ndf * 4
            nn.Conv2d(ndf * 4, ndf * 8, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 8),
            nn.LeakyReLU(0.2, inplace=True),  # 16 * 16 * ndf * 8
            nn.Conv2d(ndf * 8, ndf * 16, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 16),
            nn.LeakyReLU(0.2, inplace=True),  # 8 * 8 * ndf * 16
            nn.Conv2d(ndf * 16, ndf * 32, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 32),
            nn.LeakyReLU(0.2, inplace=True),  # 4 * 4 * ndf * 32
            conv3x3(ndf * 32, ndf * 16),
            nn.BatchNorm2d(ndf * 16),
            nn.LeakyReLU(0.2, inplace=True),  # 4 * 4 * ndf * 16
            conv3x3(ndf * 16, ndf * 8),
            nn.BatchNorm2d(ndf * 8),
            nn.LeakyReLU(0.2, inplace=True)  # 4 * 4 * ndf * 8
        )

        if self.bcondition:
            self.outlogits = nn.Sequential(
                conv3x3(ndf * 8 + nef, ndf * 8),
                nn.BatchNorm2d(ndf * 8),
                nn.LeakyReLU(0.2, inplace=True),
                nn.Conv2d(ndf * 8, 1, kernel_size=4, stride=4),
                nn.Sigmoid())
        else:
            self.outlogits = nn.Sequential(
                nn.Conv2d(ndf * 8, 1, kernel_size=4, stride=4),
                nn.Sigmoid())

    def forward(self, image, condition=None):
        image_feature = self.encode_image(image)
        if self.bcondition:
            condition = condition.view(-1, self.nef, 1, 1)
            condition = condition.repeat(1, 1, 4, 4)
            # state size (ngf+nef) x 4 x 4
            code = torch.cat((image_feature, condition), 1)
        else:
            code = image_feature

        output = self.outlogits(code)
        return output.view(-1)
