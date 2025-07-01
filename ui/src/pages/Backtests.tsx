import React from 'react';
import { Typography, Box } from '@mui/material';

const Backtests: React.FC = () => {
  return (
    <Box>
      <Typography variant="h4" gutterBottom>
        Backtests
      </Typography>
      <Typography variant="body1">
        Backtest results and strategy performance will be displayed here.
      </Typography>
    </Box>
  );
};

export default Backtests; 