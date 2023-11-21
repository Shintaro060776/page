import React from 'react';

const BackLink = ({ href, text }) => {
    return (
        <a href={href} style={{ color: 'white', textDecoration: 'none' }}>
            {text}
        </a>
    );
};

export default BackLink;