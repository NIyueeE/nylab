import React, { useState } from 'react';
import { Upload, Button, Select, Card, Progress, notification } from 'antd';
import { InboxOutlined } from '@ant-design/icons';
import { startTraining, getTrainingProgress } from '../services/api';

const { Dragger } = Upload;
const { Option } = Select;

const TrainPage = () => {
  const [file, setFile] = useState(null);
  const [modelType, setModelType] = useState('random_forest');
  const [training, setTraining] = useState(false);
  const [progress, setProgress] = useState(0);
  const [runId, setRunId] = useState(null);

  const handleUpload = (info) => {
    const { file } = info;
    if (file.status === 'uploading') {
      setFile(null); // 重置文件状态
      return;
    }

    if (file.status === 'done') {
      // 验证文件有效性
      if (file.originFileObj) {
        setFile(file.originFileObj);
        notification.success({ message: '文件上传成功' });
      } else {
        notification.error({ message: '文件无效' });
      }
    }
    else if (file.status === 'error') {
      notification.error({ message: '上传失败', description: file.response?.message || '未知错误' });
      setFile(null);
    }
    else if (file.status === 'removed') {
      setFile(null);
    }
  };

  const handleStartTraining = async () => {
    if (!file) {
      notification.warning({ message: '请先上传数据集' });
      return;
    }

    setTraining(true);

    try {
      const formData = new FormData();
      formData.append('dataset', file);
      formData.append('model_type', modelType);

      const response = await startTraining(formData);
      setRunId(response.run_id);

      // 轮询获取进度
      const interval = setInterval(async () => {
        const progressData = await getTrainingProgress(response.run_id);
        setProgress(progressData.progress);

        if (progressData.status === 'completed' || progressData.status === 'failed') {
          clearInterval(interval);
          setTraining(false);

          if (progressData.status === 'completed') {
            notification.success({
              message: '训练完成',
              description: `准确率: ${progressData.accuracy}`
            });
          } else {
            notification.error({
              message: '训练失败',
              description: progressData.message
            });
          }
        }
      }, 2000);
    } catch (error) {
      notification.error({ message: '训练启动失败', description: error.message });
      setTraining(false);
    }
  };

  return (
    <div className="container">
      <Card title="模型训练平台" style={{ maxWidth: 800, margin: '0 auto' }}>
        <Dragger
          name="dataset"
          multiple={false}
          accept=".csv,.parquet"
          beforeUpload={() => false}
          onChange={handleUpload}
          showUploadList={false}
        >
          <p className="ant-upload-drag-icon">
            <InboxOutlined />
          </p>
          <p>点击或拖拽文件到此处上传数据集</p>
          {file && <p>已选择: {file.name}</p>}
        </Dragger>

        <div style={{ margin: '24px 0' }}>
          <Select
            value={modelType}
            onChange={setModelType}
            style={{ width: '100%' }}
            disabled={training}
          >
            <Option value="random_forest">随机森林</Option>
            <Option value="xgboost">XGBoost</Option>
            <Option value="svm">支持向量机</Option>
            <Option value="cnn">卷积神经网络</Option>
          </Select>
        </div>

        <Button
          type="primary"
          onClick={handleStartTraining}
          loading={training}
          disabled={!file || training}
          block
        >
          开始训练
        </Button>

        {progress > 0 && (
          <div style={{ marginTop: 24 }}>
            <Progress percent={progress} status="active" />
            <div style={{ textAlign: 'center', marginTop: 8 }}>
              训练中... {progress}%
            </div>
          </div>
        )}

        {runId && !training && (
          <div style={{ marginTop: 24, textAlign: 'center' }}>
            <Button
              type="link"
              href={`http://${window.location.hostname}:5000/#/experiments/0/runs/${runId}`}
              target="_blank"
            >
              在MLflow中查看训练详情
            </Button>
          </div>
        )}
      </Card>
    </div>
  );
};

export default TrainPage;