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
import torch.optim as optim

from BNUDataset.BNU_Dataload import BNUDataset

def train_model(EEG_model, EYE_model, Emotion_model, cross_et, cross_emo, alignment_model, train_loader, test_loader, criterion, EYE_criterion, Emotion_criterion, optimizer, EYE_optimizer, Emotion_optimizer, device, num_epochs):
    EEG_model = EEG_model.to(device)

    # 设置保存目录
    save_dir = 'best_model'  # 文件夹
    # 如果文件夹不存在，创建它
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    # 设置完整的保存路径，确保包含文件名和扩展名
    save_path = os.path.join(save_dir, 'best_model.pth')

    best_accuracy = 40
    for epoch in range(num_epochs):
        EEG_model.train()
        running_loss = 0.0
        correct_predictions = 0
        all_preds = []
        all_labels = []

        # 用于统计每个类别的正确预测数和总数
        class_correct = [0] * 3  # 3个类别
        class_total = [0] * 3

        for batch in train_loader:
            # print('batch', batch)
            eeg_batch = batch['eeg'].cuda()
            eye_batch = batch['eye'].cuda()
            emotion_batch = batch['emotion'].cuda()
            targets = batch['label'].cuda()


            optimizer.zero_grad()
            EYE_optimizer.zero_grad()
            Emotion_optimizer.zero_grad()
            EEG_outputs, EEG_feature = EEG_model(eeg_batch)
            EYE_outputs, EYE_feature = EYE_model(eye_batch)
            Emotion_outputs, Emotion_feature = Emotion_model(emotion_batch)

            outputs, feat = alignment_model(EEG_feature, EYE_feature, Emotion_feature)

            EEG_loss = criterion(EEG_outputs, targets)
            EYE_loss = EYE_criterion(EYE_outputs, targets)
            Emotion_loss = Emotion_criterion(Emotion_outputs, targets)

            fusion_loss = criterion(outputs, targets)

            loss = (EEG_loss + EYE_loss + Emotion_loss + fusion_loss)/4

            loss.backward()

            optimizer.step()
            EYE_optimizer.step()

            running_loss += loss.item()

            _, predicted = outputs.max(1)
            correct_predictions += (predicted == targets).sum().item()

            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(targets.cpu().numpy())

            # 计算每个类别的准确度
            for i in range(targets.size(0)):
                label = targets[i]
                class_total[label] += 1
                if predicted[i] == label:
                    class_correct[label] += 1

        avg_loss = running_loss / len(train_loader)
        accuracy = correct_predictions / len(train_loader.dataset) * 100

        print(f"Epoch [{epoch + 1}/{num_epochs}], Loss: {avg_loss:.4f}, Accuracy: {accuracy:.2f}%")

        EEG_model.eval()
        EYE_model.eval()
        Emotion_model.eval()
        alignment_model.eval()

        test_correct = 0
        test_total = 0
        with torch.no_grad():
            for test_batch in test_loader:
                eeg = test_batch['eeg'].to(device)
                eye = test_batch['eye'].to(device)
                emotion = test_batch['emotion'].to(device)
                targets = test_batch['label'].to(device)

                EEG_outputs, EEG_feature = EEG_model(eeg)
                EYE_outputs, EYE_feature = EYE_model(eye)
                Emotion_outputs, Emotion_feature = Emotion_model(emotion)
                outputs, feat = alignment_model(EEG_feature, EYE_feature, Emotion_feature)

                _, predicted = outputs.max(1)
                test_total += targets.size(0)
                test_correct += (predicted == targets).sum().item()

        epoch_accuracy = 100 * test_correct / test_total
        print(f"Test Accuracy: {epoch_accuracy:.2f}%")

        # 保存最佳模型
        if epoch_accuracy > best_accuracy:
            best_accuracy = epoch_accuracy
            torch.save({
                'EEG_model': EEG_model.state_dict(),
                'EYE_model': EYE_model.state_dict(),
                'Emotion_model': Emotion_model.state_dict(),
                'alignment_model': alignment_model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'epoch': epoch,
                'best_accuracy': best_accuracy,
            }, save_path)
            print(f"Saved new best model with accuracy {best_accuracy:.2f}%")

    print(f"Training complete! Best accuracy: {best_accuracy:.2f}%")

# ============================== 模型评估 ==================================== #
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

    with torch.no_grad():
        for batch in test_loader:

            eeg_batch = batch['eeg'].cuda()
            eye_batch = batch['eye'].cuda()
            emotion_batch = batch['emotion'].cuda()
            targets = batch['label'].cuda()

            EEG_outputs, EEG_feature = EEG_model(eeg_batch)
            EYE_outputs, EYE_feature = EYE_model(eye_batch)
            Emotion_outputs, Emotion_feature = Emotion_model(emotion_batch)

            outputs, feat = alignment_model(EEG_feature, EYE_feature, Emotion_feature)

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

    precision = precision_score(all_labels, all_preds, average='weighted')
    recall = recall_score(all_labels, all_preds, average='weighted')
    f1 = f1_score(all_labels, all_preds, average='weighted')
    conf_matrix = confusion_matrix(all_labels, all_preds)

    # 计算整体准确度
    overall_accuracy = correct_predictions / total_samples * 100

    print("Evaluation Metrics:")
    print(f"Accuracy: {overall_accuracy}, Precision: {precision:.4f}, Recall: {recall:.4f}, F1 Score: {f1:.4f}")

    # 输出每个类别的准确度
    for i in range(3):
        class_accuracy = 100 * class_correct[i] / (class_total[i]) if class_total[i] > 0 else 0
        if class_accuracy > 100:
            class_accuracy = 100
        print(f"Class {i} Accuracy: {class_accuracy:.2f}%")

    # 输出整体准确度
    print(f"Overall Accuracy: {overall_accuracy:.2f}%")

    return overall_accuracy  # 返回整体准确度，以便用于模型保存

# ============================== 主程序 ==================================== #
if __name__ == "__main__":
    asd_path = r"E:\dataset\CCNU\preprocessing_data\asd"
    td_path = r"E:\dataset\CCNU\preprocessing_data\td"

    eeg_dataset = EEGDataset(asd_path, td_path, duration=2.0, sfreq=256)

    # 创建训练和测试 DataLoader
    train_data, train_labels = eeg_dataset.get_train_data()
    test_data, test_labels = eeg_dataset.get_test_data()

    # 构建 DataLoader
    train_loader = DataLoader(MultiModalDataset(train_data), batch_size=16, shuffle=True)
    test_loader = DataLoader(MultiModalDataset(test_data), batch_size=16, shuffle=True)

    # 模型参数
    num_channels = train_data[0]['eeg'].shape[0]  # EEG 通道数
    time_points = train_data[0]['eeg'].shape[1]  # 时间步数

    # 初始化 Transformer 模型
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    EEG_model = EEGTransformer(num_channels=num_channels, time_points=time_points)
    EYE_model = MLPModel(input_size=4, hidden_size=64, num_classes=3).cuda()
    Emotion_model = FaceEmotionFeatureExtractor(output_dim=128)

    cross_et = CrossAttention(embed_dim=128)  # ET对齐EEG
    cross_emo = CrossAttention(embed_dim=128)  # Emotion对齐EEG

    alignment_model = EEG_ET_Emotion_Fusion()

    EEG_criterion = nn.CrossEntropyLoss().cuda()
    EYE_criterion = nn.CrossEntropyLoss().cuda()
    Emotion_criterion = nn.CrossEntropyLoss().cuda()
    EEG_optimizer = torch.optim.AdamW(EEG_model.parameters(), lr=1e-4, weight_decay=1e-5)
    EYE_optimizer = torch.optim.AdamW(EYE_model.parameters(), lr=1e-4, weight_decay=1e-5)
    Emotion_optimizer = torch.optim.AdamW(EYE_model.parameters(), lr=1e-4, weight_decay=1e-5)


    print('Begin training...')
    # 训练模型
    train_model(EEG_model, EYE_model, Emotion_model, cross_et, cross_emo, alignment_model, train_loader, test_loader, EEG_criterion, EYE_criterion, Emotion_criterion,
                EEG_optimizer, EYE_optimizer, Emotion_optimizer, device, num_epochs=500)

    print('Training complete!')

    print('Begin testing...')
    # 测试模型
    evaluate_model(EEG_model, EYE_model, Emotion_model, alignment_model, test_loader, device)
    print('Testing complete!')