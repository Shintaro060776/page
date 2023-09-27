import axios from 'axios';

const API_ENDPOINT = 'https://your-api-gateway-url/posts';

export const getAllPosts = async () => {
    try {
        const response = await axios.get(API_ENDPOINT);
        return response.data;
    } catch (error) {
        console.error("Error fetching posts:", error);
        throw error;
    }
};

export const createPost = async (post) => {
    try {
        const response = await axios.post(API_ENDPOINT, post);
        return response.data;
    } catch (error) {
        console.error("Error creating post:", error);
        throw error;
    }
};

export const updatePost = async (postId, updatedData) => {
    try {
        const response = await axios.put(`${API_ENDPOINT}/${postId}`, updatedData);
        return response.data;
    } catch (error) {
        console.error("Error updating post:", error);
        throw error;
    }
};

export const deletePost = async (postId) => {
    try {
        const response = await axios.delete(`${API_ENDPOINT}/${postId}`);
        return response.data;
    } catch (error) {
        console.error("Error deleting post:", error);
        throw error;
    }
};