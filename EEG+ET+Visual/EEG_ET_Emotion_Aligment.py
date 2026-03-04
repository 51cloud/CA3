import torch
import torch.nn as nn

class EEG_ET_Emotion_Fusion(nn.Module):
    def __init__(self, eeg_dim=64, eye_dim=64, emotion_input_dim=48, num_classes=3):
        super(EEG_ET_Emotion_Fusion, self).__init__()

        # 投影 emotion 特征：48 → 64
        self.emotion_proj = nn.Linear(emotion_input_dim, eeg_dim).cuda()

        # Cross-Attention: EEG ← EYE
        self.cross_attn_et = nn.MultiheadAttention(embed_dim=eeg_dim, num_heads=4, batch_first=True).cuda()

        # Cross-Attention: EEG ← Emotion
        self.cross_attn_emotion = nn.MultiheadAttention(embed_dim=eeg_dim, num_heads=4, batch_first=True).cuda()

        # 分类器
        self.classifier = nn.Sequential(
            nn.Linear(eeg_dim, 64),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, num_classes)
        ).cuda()

    def GAP(self, r1, r2, alpha):
        # print('alpha1:', alpha)
        u = Uncertain(alpha).__next__()
        # print('u:', u)
        # (1-u1)/(3-u1-u2-u3)
        u1 = u[0]
        u2 = u[1]
        sum = 2 - (u1 + u2)
        r = (1 - u1) / sum * r1 + (1 - u2) / sum * r2
        '''
        print('sum: ', sum)
        print('(1-u[0])/sum: ', (1 - u[0]) / sum)
        print('(1-u[1])/sum: ', (1 - u[1]) / sum)
        print('(1-u[2])/sum: ', (1 - u[2]) / sum)

        b, c, t, w, h = r1.size()
        r = np.empty((b, c, t, w, h))
        r1 = r1.cpu().detach().numpy()
        r2 = r2.cpu().detach().numpy()
        r3 = r3.cpu().detach().numpy()
        for i in range(b):
            u1 = u[0][i][0].cpu().detach().numpy()
            u2 = u[1][i][0].cpu().detach().numpy()
            u3 = u[2][i][0].cpu().detach().numpy()
            sum = 3 - u1 - u2 - u3
            # print('r1[i]:', r1[i])
            # print('u1:', u1)
            print('sum: ', sum)
            print('((1-u1)/sum: ', ((1-u1)/sum))
            print('((1-u2)/sum: ', ((1-u2)/sum))
            print('((1-u3)/sum: ', ((1-u3)/sum))
            r[i] = r1[i]*((1-u1)/sum) + r2[i]*((1-u2)/sum) + r3[i]*((1-u3)/sum)
        r = torch.from_numpy(r).cuda()
        r = torch.tensor(r, dtype=torch.float32)
        return r

        uu = dict()
        for v in range(params.num_views):
            uu[v] = torch.sum(u[v], dim=0, keepdim=True)/float(params.batch_size)
        print('uu:', uu)
        u1 = uu[0]
        u2 = uu[1]
        u3 = uu[2]
        sum = 3-u1-u2-u3
        '''
        # u1 = 0.5
        # u2 = 0.4
        # u3 = 0.3
        # sum = 3-(u1+u2+u3)
        # r = r1*((1-u1)/sum) + r2*((1-u2)/sum) + r3*((1-u3)/sum)
        # r = (r1 + r2) / 2
        return r

    def forward(self, eeg_feat, eye_feat, emotion_feat):
        """
        eeg_feat: [T, B, 64]  → 转换为 [B, T, 64]
        eye_feat: [B, 64]     → 扩展为 [B, T, 64]
        emotion_feat: [B, 512, 3, 4, 4] → reshape + proj → [B, T, 64]
        """
        # EEG: [T, B, D] → [B, T, D]
        eeg_feat = eeg_feat.permute(1, 0, 2)  # [B, T, D]
        B, T, D = eeg_feat.shape

        # EYE: [B, D] → [B, T, D]
        eye_feat = eye_feat.unsqueeze(1).expand(-1, T, -1)  # repeat along time

        # Emotion: [B, 512, 3, 4, 4] → [B, 512, 48]
        emotion_feat = emotion_feat.view(B, 512, -1)  # flatten per-frame → [B, 512, 48]

        # 投影 Emotion 到 [B, 512, 64]
        emotion_feat = self.emotion_proj(emotion_feat)  # [B, 512, 64]

        # 对 Emotion 特征进行对齐（若不足 T 帧，补零；若多于 T 帧，截断）
        if emotion_feat.size(1) < T:
            pad_len = T - emotion_feat.size(1)
            pad = torch.zeros(B, pad_len, D, device=emotion_feat.device)
            emotion_feat = torch.cat([emotion_feat, pad], dim=1)  # [B, T, 64]
        elif emotion_feat.size(1) > T:
            emotion_feat = emotion_feat[:, :T, :]  # 截断到 T

        # Cross-Attention: EEG ← EYE
        eeg_et_aligned, _ = self.cross_attn_et(query=eeg_feat, key=eye_feat, value=eye_feat)

        # Cross-Attention: EEG ← Emotion
        eeg_emotion_aligned, _ = self.cross_attn_emotion(query=eeg_feat, key=emotion_feat, value=emotion_feat)

        """
            _, alpha1, _ = self.head(x)

            _, alpha2, _ = self.head(x2)

            alphas = [alpha1[0], alpha2[0]]

            x1 = GAP(x[0], x2[0], alphas, self.cfg)
        """

        # 特征融合
        fused_feat = eeg_feat + eeg_et_aligned + eeg_emotion_aligned  # [B, T, 64]

        # # 池化（平均）
        fused_feat = fused_feat.mean(dim=1)  # [B, 64]

        eeg_et_aligned_feat = eeg_et_aligned.mean(dim=1)  # [B, 64]

        eeg_emotion_aligned_feat = eeg_emotion_aligned.mean(dim=1)  # [B, 64]

        # 分类
        eeg_et_aligned_feat_out = self.classifier(eeg_et_aligned_feat)
        eeg_emotion_aligned_out = self.classifier(eeg_emotion_aligned_feat)  # [B, num_classes]

        # alpha = [eeg_et_aligned_feat_out, eeg_emotion_aligned_out]
        #
        # eeg_et_emotion_out = self.GAP(eeg_et_aligned_feat, eeg_emotion_aligned_feat, alpha)
        #
        # out = self.classifier(eeg_et_emotion_out)

        out = (eeg_et_aligned_feat_out + eeg_emotion_aligned_out) / 2

        # out = self.classifier(fused_feat)

        return out, fused_feat

class Uncertain():
    def __init__(self, alpha):
        super(Uncertain, self).__init__()
        self.views = 1
        # self.num_classes = params.num_classes
        self.num_classes = 3
        self.alpha = alpha

    def DS_Combin(self, alpha):
        # alpha1 = alpha1.cpu()
        # alpha1 = nn.BatchNorm1d(self.num_classes)(alpha1)
        # alpha1 = nn.ReLU()(alpha1).cuda()
        # print('alpha1: ', alpha1)
        # alpha2 = alpha2.cpu().detach().numpy()
        # alpha2 = nn.BatchNorm1d(self.num_classes)(alpha2)
        # alpha2 = nn.ReLU()(alpha2).cuda()
        # print('alpha2: ', alpha2)
        b, S, E, u = dict(), dict(), dict(), dict()
        for v in range(len(self.alpha)):
            S[v] = torch.sum(alpha[v], dim=-1, keepdim=True)
            E[v] = alpha[v] - 1
            b[v] = E[v] / (S[v].expand(E[v].shape))
            u[v] = self.num_classes / S[v]
            # print('u[v]:', u[v])
        return u

    def __iter__(self):
        return self

    def __next__(self):
        alpha = self.alpha
        # print('alpha:', alpha)
        for v_num in range(len(alpha)):
            # step two
            alpha[v_num] = alpha[v_num] + 1
        # print('alphs[0]: ', alpha[0])
        # print('alphs[1]: ', alpha)
        u = self.DS_Combin(alpha)

        return u

# @torchsnooper.snoop()
class DAL_regularizer(nn.Module):
    '''
    Disentangled Feature Learning module in paper.
    '''

    def __init__(self, ps, ns):
        super().__init__()
        self.discrimintor = nn.Sequential(nn.Linear(4096, 2048)
                                          , nn.ReLU(inplace=True)
                                          , nn.Linear(2048, 2048)
                                          , nn.ReLU(inplace=True)
                                          , nn.Linear(2048, 1)
                                          , nn.Sigmoid()).cuda()
        self.ps = ps.cuda()
        self.ns = ns.cuda()

    def __next__(self):
        # ps1 = self.ps.mean(dim=-1).mean(dim=-1).mean(dim=-1)
        # ps1 = ps1.cuda()
        # ns1 = self.ns.mean(dim=-1).mean(dim=-1).mean(dim=-1)
        # ns1 = ns1.cuda()
        ps_scores = self.discrimintor(self.ps)
        # torch.backends.cudnn.enabled = False
        ns_scores = self.discrimintor(self.ns)

        return ps_scores, ns_scores



# # 测试代码
# if __name__ == '__main__':
#     model = EEG_ET_Emotion_Fusion().cuda()
#
#     eeg = torch.randn(513, 16, 64).cuda()               # [T=513, B=16, D=64]
#     eye = torch.randn(16, 64).cuda()                    # [B, D=64]
#     emotion = torch.randn(16, 512, 3, 4, 4).cuda()       # [B, T=512, C=3, H=4, W=4]
#
#     output = model(eeg, eye, emotion)
#     print(output.shape, output)  # ➤ torch.Size([16, 3])
