import axios from 'axios';

const API_URL = process.env.REACT_APP_API_URL || 'http://localhost:8000';

export const startTraining = async (formData) => {
  const response = await axios.post(`${API_URL}/api/train`, formData, {
    headers: {
      'Content-Type': 'multipart/form-data'
    }
  });
  return response.data;
};

export const getTrainingProgress = async (runId) => {
  const response = await axios.get(`${API_URL}/api/progress/${runId}`);
  return response.data;
};