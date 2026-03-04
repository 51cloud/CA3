import os
import mne
import numpy as np
import torch
from torch.utils.data import Dataset
from sklearn.model_selection import train_test_split
import pandas as pd
from sklearn.cluster import KMeans
import cv2
import h5py

# from EYE.eyeVisual import video_path

# person_id 与 level 的映射表
person_id_to_level = {
    '05': 1, '06': 3, '07': 3, '08': 2, '09': 1, '10': 2,
    '11': 3, '12': 3, '13': 2, '14': 1, '15': 2, '16': 1, '17': 3, '18': 3, '19': 1, '20': 3,
    '21': 2, '22': 1, '23': 2, '24': 1, '25': 3, '26': 1, '28': 1, '29': 3, '30': 1,
    '31': 3, '32': 2, '33': 1, '34': 3, '35': 3
}

class EEGDataset(Dataset):
    def __init__(self, asd_root_path, td_root_path, duration=2.0, sfreq=256, test_size=0.3):
        self.aoi_root_dir = r'E:\dataset\CCNU\preprocessing_data\AOI\asd'
        self.duration = duration
        self.sfreq = sfreq

        # Load ASD data
        self.all_data, self.all_labels = self.load_folder(asd_root_path, is_td=False)

        # Standardize the data (EEG and eye features)
        data = self.all_data
        # print('data:', data)
        for item in self.all_data:
            item['eeg'] = (item['eeg'] - np.mean(item['eeg'], axis=(0, 1), keepdims=True)) / \
                          (np.std(item['eeg'], axis=(0, 1), keepdims=True) + 1e-6)
            item['eye'] = (item['eye'] - np.mean(item['eye'], axis=0)) / (np.std(item['eye'], axis=0) + 1e-6)

        # Split into training and testing sets
        self.train_data, self.test_data, self.train_labels, self.test_labels = train_test_split(
            data, self.all_labels, test_size=test_size, random_state=42, stratify=self.all_labels
        )

        # self.train_data, self.train_labels, self.test_data, self.test_labels = self.load_folder(asd_root_path, is_td=False)
        #
        # for item in self.train_data:
        #     item['eeg'] = (item['eeg'] - np.mean(item['eeg'], axis=(0, 1), keepdims=True)) / \
        #                   (np.std(item['eeg'], axis=(0, 1), keepdims=True) + 1e-6)
        #     item['eye'] = (item['eye'] - np.mean(item['eye'], axis=0)) / (np.std(item['eye'], axis=0) + 1e-6)
        #
        # for item in self.test_data:
        #     item['eeg'] = (item['eeg'] - np.mean(item['eeg'], axis=(0, 1), keepdims=True)) / \
        #                   (np.std(item['eeg'], axis=(0, 1), keepdims=True) + 1e-6)
        #     item['eye'] = (item['eye'] - np.mean(item['eye'], axis=0)) / (np.std(item['eye'], axis=0) + 1e-6)

    def load_folder(self, root_path, is_td=False):
        all_data = []
        all_labels = []

        train_data = []
        train_labels = []

        test_data = []
        test_labels = []
        train_person_list = ["05","08","09","10","12","14","16","19","26","13","15","22","06","07","11","17","18","20","21","25","29"]
        test_person_list = ["18","30","33","23","24","31","32","34","35"]

        Positive_list = ['Pleasure', 'Surprise']
        Negative_list = ['Sadness', 'Anger', 'Fear', 'Disgust']
        Neutral_list = ['Quiet']

        Positive_data = []  # Pleasure, Surprise
        Positive_labels = []

        Negative_data = []  # Sadness, Anger, Fear, Disgust
        Negative_labels = []

        Neutral_data = []  # Quiet
        Neutral_labels = []

        for person_id in os.listdir(root_path):

            person_path = os.path.join(root_path, person_id, 'EEG_Segments')

            if os.path.isdir(person_path):
                if is_td:
                    level = 0
                else:
                    level = person_id_to_level.get(person_id, 1) - 1

                for file_name in os.listdir(person_path):
                    if file_name.endswith('.set'):
                        file_path = os.path.join(person_path, file_name)
                        print(f"Loading {file_path} (person_id: {person_id}, level: {level}) ...")

                        eeg_data = self.load_eeg_file(file_path)
                        if eeg_data is not None:

                            # Extract emotion info from file name
                            emotion = file_name.split('.')[0]
                            if len(emotion.split('_')) == 2:
                                emotion = emotion.split('_')[1]

                            # Read AOI data
                            if emotion != 'Quiet':
                                aoi_file = os.path.join(self.aoi_root_dir, person_id, 'AOI_Results',
                                                        f"{emotion}_aoi_results.csv")
                                if not os.path.exists(aoi_file):
                                    aoi_features = np.zeros((eeg_data.shape[0], 4))
                                else:
                                    aoi_df = pd.read_csv(aoi_file)
                                    aoi_df.columns = aoi_df.columns.str.strip()
                                    aoi_df['TimeWindow'] = (aoi_df['Frame'] // 50).astype(int)
                                    grouped = aoi_df.groupby('TimeWindow')

                                    eye_data = []
                                    for time_window, group in grouped:
                                        aoi_features = self.extract_aoi_features(group)
                                        eye_data.append(aoi_features)
                                emotion_feature_name = file_name.split('.')[0] + "_faces.npy"
                                video_name = file_name.split('.')[0] + ".mp4"

                                if emotion == 'Surprise':
                                    emotion_feature_name = emotion_feature_name.split('06_')[1]
                                    video_name = video_name.split('_')[1]

                                emotion_file = os.path.join(root_path, person_id, 'VIDEO_INFO', emotion_feature_name)
                                video_path = os.path.join(root_path, person_id, 'VIDEO_Segments', video_name)
                                cap = cv2.VideoCapture(video_path)
                                if not cap.isOpened():
                                    print(f"无法打开视频文件: {video_path}")
                                    return

                                fps = cap.get(cv2.CAP_PROP_FPS)
                                print('fps:', fps)

                                # 读取视频人脸特征
                                face_segments = []
                                if os.path.exists(emotion_file):
                                    try:
                                        emotion_data = np.load(emotion_file, allow_pickle=False)
                                    except Exception as e:
                                        print("[SKIP BAD NPY]", emotion_file, repr(e))
                                        continue
                                    segment_frame_count = int(fps * 2)  # 2秒一段
                                    total_frames = emotion_data.shape[0]
                                    num_segments = total_frames // segment_frame_count + (
                                        1 if total_frames % segment_frame_count > 0 else 0)

                                    for i in range(num_segments):
                                        start = i * segment_frame_count
                                        end = min(start + segment_frame_count, total_frames)
                                        segment = emotion_data[start:end]
                                        # face_feature = np.mean(segment, axis=(0, 1, 2))  # 平均池化特征
                                        face_segments.append(segment)
                                else:
                                    print(f"人脸特征文件不存在: {emotion_file}")
                                if eeg_data is not None and eye_data:
                                    min_windows = min(eeg_data.shape[0], len(eye_data), len(face_segments))
                                    eeg_data = eeg_data[:min_windows]
                                    aoi_features_list = eye_data[:min_windows]
                                    face_segments = face_segments[:min_windows]

                                    # Store EEG, AOI, and label data
                                    for i in range(min_windows):
                                        emotion = self.resize_segment_to_fixed_frames(face_segments[i])
                                        # Save EEG, AOI, and label into lists
                                        all_data.append({
                                            'eeg': eeg_data[i],
                                            'eye': aoi_features_list[i],
                                            'emotion': emotion,
                                            'label': level
                                        })
                                        all_labels.append(level)

                                        if person_id in train_person_list:
                                            train_data.append({
                                                'eeg': eeg_data[i],
                                                'eye': aoi_features_list[i],
                                                'label': level
                                            })
                                            train_labels.append(level)
                                        else:
                                            test_data.append({
                                                'eeg': eeg_data[i],
                                                'eye': aoi_features_list[i],
                                                'label': level
                                            })
                                            test_labels.append(level)

                                        if emotion in Positive_list:
                                            Positive_data.append({
                                                'eeg': eeg_data[i],
                                                'eye': aoi_features_list[i],
                                                'face': face_segments[i],
                                                'label': level
                                            })
                                            Positive_labels.append(level)
                                        elif emotion in Negative_list:
                                            Negative_data.append({
                                                'eeg': eeg_data[i],
                                                'eye': aoi_features_list[i],
                                                'face': face_segments[i],
                                                'label': level
                                            })
                                            Negative_labels.append(level)
                                        else :
                                            Neutral_data.append({
                                                'eeg': eeg_data[i],
                                                'eye': aoi_features_list[i],
                                                'label': level
                                            })
                                            Neutral_labels.append(level)
                            else:
                                aoi_features = np.zeros((eeg_data.shape[0], 4))
                                emotion_feature_name = file_name.split('.')[0] + "_faces.npy"
                                emotion_file = os.path.join(root_path, person_id, 'VIDEO_INFO', emotion_feature_name)
                                video_name = file_name.split('.')[0] + ".mp4"
                                video_path = os.path.join(root_path, person_id, 'VIDEO_Segments', video_name)
                                cap = cv2.VideoCapture(video_path)
                                if not cap.isOpened():
                                    print(f"无法打开视频文件: {video_path}")
                                    return

                                fps = cap.get(cv2.CAP_PROP_FPS)

                                # 读取视频人脸特征
                                face_segments = []
                                if os.path.exists(emotion_file):
                                    emotion_data = np.load(emotion_file)  # shape: (num_frames, H, W, 3)
                                    segment_frame_count = int(fps * 2)  # 2秒一段
                                    total_frames = emotion_data.shape[0]
                                    num_segments = total_frames // segment_frame_count + (
                                        1 if total_frames % segment_frame_count > 0 else 0)

                                    for i in range(num_segments):
                                        start = i * segment_frame_count
                                        end = min(start + segment_frame_count, total_frames)
                                        segment = emotion_data[start:end]
                                        face_segments.append(segment)
                                else:
                                    print(f"人脸特征文件不存在: {emotion_file}")
                                if eeg_data is not None and eye_data:
                                    min_windows = min(eeg_data.shape[0], len(eye_data), len(face_segments))
                                    eeg_data = eeg_data[:min_windows]
                                    aoi_features_list = eye_data[:min_windows]
                                    face_segments = face_segments[:min_windows]

                                    # Store EEG, AOI, and label data
                                    for i in range(min_windows):
                                        emotion = self.resize_segment_to_fixed_frames(face_segments[i])
                                        all_data.append({
                                            'eeg': eeg_data[i],
                                            'eye': aoi_features_list[i],
                                            'emotion': emotion,
                                            'label': level
                                        })
                                        all_labels.append(level)
                                        # print(aaa)
        cluster_labels = self.label_Encapsulation(all_data)

        # Assign cluster labels to each sample
        for idx, data_dict in enumerate(all_data):
            data_dict['label'] = cluster_labels[idx]
        return all_data, all_labels

    def load_eeg_file(self, file_path):
        old_base = r'E:\dataset\CCNU\preprocessing_data\asd'
        new_base = r'E:\dataset\CCNU\preprocessing_data\asd_converted'

        # 正确获取相对路径
        relative_path = os.path.relpath(file_path, old_base).split('.')[0] + '_converted.set'
        new_file_path = os.path.join(new_base, relative_path)

        if new_file_path not in os.listdir(new_base):
            new_file_path = file_path
        if os.path.getsize(new_file_path) == 0:
            print(f"Warning: File {new_file_path} is empty, skipping.")
            return None

        try:
            raw = mne.io.read_raw_eeglab(new_file_path, preload=True)
            if raw.n_times == 0:
                print(f"Warning: No data found in {new_file_path}, skipping this file.")
                return None

            if raw.annotations is not None:
                new_annotations = mne.Annotations(onset=raw.annotations.onset,
                                                  duration=raw.annotations.duration,
                                                  description=raw.annotations.description)
                new_annotations = new_annotations.crop(0, raw.times[-1])
                raw.set_annotations(new_annotations)

            events = mne.make_fixed_length_events(raw, duration=self.duration)
            epochs = mne.Epochs(raw, events, tmin=0, tmax=self.duration, baseline=None, preload=True,
                                reject_by_annotation=True)

            return epochs.get_data(copy=True)

        except Exception as e:
            print(f"Error reading file {new_file_path}: {e}")
            return None

    def label_Encapsulation(self, all_data):
        all_eeg_data = np.vstack([features for data_dict in all_data for features in [data_dict['eeg']]])
        kmeans = KMeans(n_clusters=3, random_state=42)
        cluster_labels = kmeans.fit_predict(all_eeg_data)
        return cluster_labels

    def extract_aoi_features(self, aoi_df):
        total_gaze_points = len(aoi_df)
        aoi_switch_count = aoi_df['Track_ID'].nunique()

        aoi_df['Duration'] = 0.04
        aoi_df = aoi_df.sort_values(by=['Frame'])

        durations = []
        start_frame = aoi_df['Frame'].iloc[0]
        start_bbox = aoi_df['Inside_BBox'].iloc[0]
        start_time_window = aoi_df['TimeWindow'].iloc[0]
        in_aoi = start_bbox

        for i in range(1, len(aoi_df)):
            if (aoi_df['Inside_BBox'].iloc[i] != in_aoi) or (aoi_df['TimeWindow'].iloc[i] != start_time_window):
                if in_aoi:
                    durations.append((aoi_df['Frame'].iloc[i - 1] - start_frame)*0.001)
                start_frame = aoi_df['Frame'].iloc[i]
                start_bbox = aoi_df['Inside_BBox'].iloc[i]
                start_time_window = aoi_df['TimeWindow'].iloc[i]
                in_aoi = start_bbox

        if in_aoi:
            durations.append((aoi_df['Frame'].iloc[-1] - start_frame) * 0.001)

        avg_duration = sum(durations) / len(durations) if durations else 0
        max_duration = max(durations) if durations else 0
        min_duration = min(durations) if durations else 0
        gaze_stability = np.std(aoi_df[['X', 'Y']].values)
        gaze_time_percentage = (aoi_df['Duration'].sum() / (len(aoi_df) * 1)) * 100
        first_fixation_time = aoi_df.iloc[0]['Frame']
        spatial_distribution = (aoi_df['X'].max() - aoi_df['X'].min()) * (aoi_df['Y'].max() - aoi_df['Y'].min())
        exploratory_behavior = np.std(aoi_df[['X', 'Y']].values)

        features = [
            aoi_switch_count,
            avg_duration,
            max_duration,
            min_duration,
        ]
        return np.array(features)

    def resize_segment_to_fixed_frames(self, segment, T_target=24):
        T_raw = segment.shape[0]
        if T_raw == T_target:
            return segment
        indices = np.linspace(0, T_raw - 1, T_target).astype(int)
        return segment[indices]

    def __len__(self):
        return len(self.train_data)

    def __getitem__(self, idx):
        eeg = torch.tensor(self.train_data[idx]['eeg'], dtype=torch.float32)
        eye = torch.tensor(self.train_data[idx]['eye'], dtype=torch.float32)
        emotion = torch.tensor(self.train_data[idx]['emotion'], dtype=torch.float32)
        label = torch.tensor(self.train_data[idx]['label'], dtype=torch.long)
        return {'eeg': eeg, 'eye': eye, 'emotion':emotion, 'label': label}

    def get_train_data(self):
        return self.train_data, self.train_labels

    def get_test_data(self):
        return self.test_data, self.test_labels
