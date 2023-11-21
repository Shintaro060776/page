import React from 'react';
import { motion } from 'framer-motion';

const RoundButton = ({ text, onClick, bgColor = '#333' }) => {
    return (
        <motion.button
            style={{
                padding: '10px 20px',
                border: 'none',
                borderRadius: '30px',
                backgroundColor: bgColor,
                color: 'white',
                cursor: 'pointer',
                outline: 'none',
                fontSize: '16px'
            }}
            whileHover={{ scale: 1.1, rotate: 5 }}
            whileTap={{ scale: 0.9 }}
            onClick={onClick}
        >
            {text}
        </motion.button>
    );
};

export default RoundButton;