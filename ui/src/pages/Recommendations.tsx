import React from 'react';
import { Typography, Box } from '@mui/material';

const Recommendations: React.FC = () => {
  return (
    <Box>
      <Typography variant="h4" gutterBottom>
        Recommendations
      </Typography>
      <Typography variant="body1">
        AI-powered betting recommendations will be displayed here.
      </Typography>
    </Box>
  );
};

export default Recommendations; 