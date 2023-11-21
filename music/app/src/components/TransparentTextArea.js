import React from 'react';

const TransparentTextArea = ({ value }) => {
    return (
        <textarea
            style={{
                width: '80%',
                height: '150px',
                backgroundColor: 'rgba(255, 255, 255, 0.1)',
                border: 'none',
                borderRadius: '10px',
                padding: '15px',
                color: 'white',
                fontSize: '16px'
            }}
            value={value}
            readOnly
        />
    );
};

export default TransparentTextArea;