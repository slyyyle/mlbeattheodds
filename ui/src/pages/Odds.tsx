import React from 'react';
import { useQuery } from '@tanstack/react-query';
import {
  Box,
  Typography,
  CircularProgress,
  Alert
} from '@mui/material';
import { DataGrid } from '@mui/x-data-grid';
import type { GridColDef } from '@mui/x-data-grid';
import { oddsApi } from '../services/api';
import type { OddsComparison } from '../types';

const Odds: React.FC = () => {
  const { data, isLoading, isError, error } = useQuery<OddsComparison[], Error>({
    queryKey: ['odds'],
    queryFn: () => oddsApi.getMoneyline(),
    staleTime: 5 * 60 * 1000,
  });

  const columns: GridColDef<OddsComparison & { id: number }>[] = [
    // eslint-disable-next-line @typescript-eslint/no-explicit-any
    { field: 'date', headerName: 'Date', width: 120, valueFormatter: (params: any) => new Date(params.value).toLocaleDateString() },
    { field: 'home_team', headerName: 'Home Team', width: 150 },
    { field: 'away_team', headerName: 'Away Team', width: 150 },
    { field: 'home_odds', headerName: 'Home Odds', width: 100 },
    { field: 'away_odds', headerName: 'Away Odds', width: 100 },
    { field: 'bookmaker', headerName: 'Bookmaker', width: 150 },
    // eslint-disable-next-line @typescript-eslint/no-explicit-any
    { field: 'vig', headerName: 'Vig', width: 100, valueFormatter: (params: any) => `${(params.value * 100).toFixed(1)}%` },
  ];

  const rows = data?.map((row, idx) => ({ id: idx, ...row })) ?? [];

  return (
    <Box>
      <Typography variant="h4" gutterBottom>
        Moneyline Odds
      </Typography>
      {isLoading ? (
        <CircularProgress />
      ) : isError ? (
        <Alert severity="error">{(error as Error).message}</Alert>
      ) : (
        <Box sx={{ height: 600, width: '100%' }}>
          <DataGrid<OddsComparison & { id: number }>
            rows={rows}
            columns={columns}
            getRowId={(row) => row.id}
            initialState={{ pagination: { paginationModel: { pageSize: 10 } } }}
            pageSizeOptions={[10, 25, 50]}
            disableRowSelectionOnClick
          />
        </Box>
      )}
    </Box>
  );
};

export default Odds; 