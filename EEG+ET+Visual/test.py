
import torch
import torch.nn as nn
from EEG.EEG_Transformer import EEGTransformer
from torch.utils.data import DataLoader, TensorDataset

from EYE.EYE_MLP import MLPModel
from dataload2 import *
from sklearn.metrics import precision_score, recall_score, f1_score, confusion_matrix, classification_report
from MultiModalDataset import MultiModalDataset
from Face.EmotionFeature import FaceEmotionFeatureExtractor
from CrossAttention import *
from EEG_ET_Emotion_Aligment import *


def load_and_evaluate(checkpoint_path,
                      EEG_model,
                      EYE_model,
                      Emotion_model,
                      alignment_model,
                      test_loader,
                      device):
    # 加载保存的checkpoint
    checkpoint = torch.load(checkpoint_path, map_location=device)

    # 加载各模型参数
    EEG_model.load_state_dict(checkpoint['EEG_model'])
    EYE_model.load_state_dict(checkpoint['EYE_model'])
    Emotion_model.load_state_dict(checkpoint['Emotion_model'])
    alignment_model.load_state_dict(checkpoint['alignment_model'])

    # 将模型移动到指定设备
    EEG_model = EEG_model.to(device)
    EYE_model = EYE_model.to(device)
    Emotion_model = Emotion_model.to(device)
    alignment_model = alignment_model.to(device)

    # 设置模型为评估模式
    EEG_model.eval()
    EYE_model.eval()
    Emotion_model.eval()
    alignment_model.eval()

    print(f"Loaded checkpoint from epoch {checkpoint['epoch']} with accuracy {checkpoint['best_accuracy']:.2f}%")

    # 执行评估
    return evaluate_model(EEG_model,
                          EYE_model,
                          Emotion_model,
                          alignment_model,
                          test_loader,
                          device)


def evaluate_model(EEG_model, EYE_model, Emotion_model, alignment_model, test_loader, device):
    EEG_model.eval()
    EYE_model.eval()
    Emotion_model.eval()
    all_preds = []
    all_labels = []
    class_correct = [0] * 3
    class_total = [0] * 3
    correct_predictions = 0  # 统计所有预测正确的样本数
    total_samples = 0        # 统计所有样本数
    features, labels = [], []
    # 创建保存目录
    save_dir = r"..\Visual\tsne_features"
    os.makedirs(save_dir, exist_ok=True)

    with torch.no_grad():
        for batch in test_loader:
            eeg_batch = batch['eeg'].cuda()
            eye_batch = batch['eye'].cuda()
            emotion_batch = batch['emotion'].cuda()
            targets = batch['label'].cuda()

            EEG_outputs, EEG_feature = EEG_model(eeg_batch)
            EYE_outputs, EYE_feature = EYE_model(eye_batch)
            Emotion_outputs, Emotion_feature = Emotion_model(emotion_batch)

            # feat = (EEG_feature + EYE_feature + Emotion_feature)/3



            # print('EYE_outputs:', EYE_outputs)
            # print(aaa)

            outputs, feat = alignment_model(EEG_feature, EYE_feature, Emotion_feature)
            # outputs = EYE_outputs
            # outputs = EEG_outputs

            features.append(feat.cpu().numpy())
            labels.append(targets.cpu().numpy())

            _, predicted = outputs.max(1)
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(targets.cpu().numpy())

            # 计算每个类别的准确度
            for i in range(targets.size(0)):
                label = targets[i]
                class_total[label] += 1
                if predicted[i] == label:
                    class_correct[label] += 1
                    correct_predictions += 1  # 每次预测正确时累加
            total_samples += targets.size(0)  # 累加每个批次的样本数

        # 合并所有batch的数据
        features = np.concatenate(features, axis=0)  # shape: (N, D)
        labels = np.concatenate(labels, axis=0)  # shape: (N,)

        # 保存为numpy格式
        np.save(os.path.join(save_dir, "features.npy"), features)
        np.save(os.path.join(save_dir, "labels.npy"), labels)
        print(f"Features saved to {save_dir}/features.npy")
        print(f"Labels saved to {save_dir}/labels.npy")

    precision = precision_score(all_labels, all_preds, average='weighted')
    recall = recall_score(all_labels, all_preds, average='weighted')
    f1 = f1_score(all_labels, all_preds, average='weighted')
    conf_matrix = confusion_matrix(all_labels, all_preds)

    # 计算整体准确度
    overall_accuracy = correct_predictions / total_samples * 100

    print("Evaluation Metrics:")
    print(f"Accuracy: {overall_accuracy}, Precision: {precision:.4f}, Recall: {recall:.4f}, F1 Score: {f1:.4f}")
    # print("Confusion Matrix:")
    # print(conf_matrix)
    # print(classification_report(all_labels, all_preds))

    # 输出每个类别的准确度
    for i in range(3):
        class_accuracy = 100 * class_correct[i] / (class_total[i]) if class_total[i] > 0 else 0
        if class_accuracy > 100:
            class_accuracy = 100
        print(f"Class {i} Accuracy: {class_accuracy:.2f}%")

    # 输出整体准确度
    print(f"Overall Accuracy: {overall_accuracy:.2f}%")

    return overall_accuracy  # 返回整体准确度，以便用于模型保存

# 使用示例
if __name__ == "__main__":
    asd_path = r"E:\dataset\CCNU\preprocessing_data\asd"
    td_path = r"E:\dataset\CCNU\preprocessing_data\td"

    # asd_path = 'E:\dataset\北师大数据集\Preprocessing_Data\EEG\ASD'

    eeg_dataset = EEGDataset(asd_path, td_path, duration=2.0, sfreq=256)
    # eeg_dataset = EEGDataset(asd_path, duration=2.0, sfreq=256)
    # eeg_dataset = BNUDataset(asd_path, duration=2.0, sfreq=256)

    # 创建训练和测试 DataLoader
    train_data, train_labels = eeg_dataset.get_train_data()
    test_data, test_labels = eeg_dataset.get_test_data()

    # train_loader = DataLoader(TensorDataset(train_data,train_labels),
    #                           batch_size=16, shuffle=True)
    # test_loader = DataLoader(TensorDataset(test_data,test_labels),
    #                          batch_size=16, shuffle=False)
    # 构建 DataLoader
    train_loader = DataLoader(MultiModalDataset(train_data), batch_size=16, shuffle=True)
    test_loader = DataLoader(MultiModalDataset(test_data), batch_size=16, shuffle=True)

    # 模型参数
    num_channels = train_data[0]['eeg'].shape[0]  # EEG 通道数
    # print('num_channels:', num_channels)
    time_points = train_data[0]['eeg'].shape[1]  # 时间步数
    # print('time_points:', time_points)

    # 初始化 Transformer 模型
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # print('device:', device)

    EEG_model = EEGTransformer(num_channels=num_channels, time_points=time_points)
    EYE_model = MLPModel(input_size=4, hidden_size=64, num_classes=3).cuda()
    Emotion_model = FaceEmotionFeatureExtractor(output_dim=128)

    cross_et = CrossAttention(embed_dim=128)  # ET对齐EEG
    cross_emo = CrossAttention(embed_dim=128)  # Emotion对齐EEG

    alignment_model = EEG_ET_Emotion_Fusion()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 加载并评估
    best_accuracy = load_and_evaluate(
        checkpoint_path='best_model/best_model.pth',
        EEG_model=EEG_model,
        EYE_model=EYE_model,
        Emotion_model=Emotion_model,
        alignment_model=alignment_model,
        test_loader=test_loader,  # 传入测试数据加载器
        device=device
    )

    print(f"Best model achieves {best_accuracy:.2f}% accuracy on test set")