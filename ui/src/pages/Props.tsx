import React from 'react';
import { useQuery } from '@tanstack/react-query';
import { Box, Typography, CircularProgress, Alert } from '@mui/material';
import { DataGrid } from '@mui/x-data-grid';
import type { GridColDef } from '@mui/x-data-grid';
import { propsApi } from '../services/api';
import type { PropBet } from '../types';

const Props: React.FC = () => {
  const { data, isLoading, isError, error } = useQuery<PropBet[], Error>({
    queryKey: ['props'],
    queryFn: () => propsApi.getAll(),
    staleTime: 5 * 60 * 1000,
  });
  
  const columns: GridColDef<PropBet>[] = [
    { field: 'game_id', headerName: 'Game ID', width: 180 },
    { field: 'player_name', headerName: 'Player', width: 180 },
    { field: 'prop_type', headerName: 'Type', width: 140 },
    { field: 'line', headerName: 'Line', width: 100, type: 'number' },
    { field: 'over_odds', headerName: 'Over Odds', width: 120, type: 'number' },
    { field: 'under_odds', headerName: 'Under Odds', width: 120, type: 'number' },
    { field: 'bookmaker', headerName: 'Bookmaker', width: 160 },
    {
      field: 'last_updated',
      headerName: 'Updated',
      width: 180,
      valueFormatter: (params: { value: string }) => new Date(params.value).toLocaleString(),
    }
  ];
  
  const rows = data ?? [];
  
  return (
    <Box>
      <Typography variant="h4" gutterBottom>
        Player Prop Bets
      </Typography>
      {isLoading ? (
        <CircularProgress />
      ) : isError ? (
        <Alert severity="error">{(error as Error).message}</Alert>
      ) : (
        <Box sx={{ height: 600, width: '100%' }}>
          <DataGrid<PropBet>
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

export default Props; 