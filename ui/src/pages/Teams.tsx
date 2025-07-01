import React from 'react';
import { useQuery } from '@tanstack/react-query';
import { Box, Typography, CircularProgress, Alert } from '@mui/material';
import { DataGrid } from '@mui/x-data-grid';
import type { GridColDef } from '@mui/x-data-grid';
import { teamsApi } from '../services/api';
import type { Team } from '../types';

const Teams: React.FC = () => {
  const { data, isLoading, isError, error } = useQuery<Team[], Error>({
    queryKey: ['teams-standings'],
    queryFn: () => teamsApi.getStandings(),
    staleTime: 5 * 60 * 1000,
  });
  
  const columns: GridColDef<Team>[] = [
    { field: 'id', headerName: 'ID', width: 80 },
    { field: 'name', headerName: 'Team', width: 200 },
    { field: 'wins', headerName: 'Wins', width: 100, type: 'number' },
    { field: 'losses', headerName: 'Losses', width: 100, type: 'number' },
    { field: 'win_percentage', headerName: 'Win %', width: 120, valueFormatter: (p: { value: number }) => (p.value * 100).toFixed(1) + '%' },
    { field: 'runs_scored', headerName: 'Runs Scored', width: 140, type: 'number' },
    { field: 'runs_allowed', headerName: 'Runs Allowed', width: 140, type: 'number' },
    { field: 'run_differential', headerName: 'Run Diff', width: 120, type: 'number' },
  ];
  
  const rows = data ?? [];
  
  return (
    <Box>
      <Typography variant="h4" gutterBottom>
        Team Standings
      </Typography>
      {isLoading ? (
        <CircularProgress />
      ) : isError ? (
        <Alert severity="error">{(error as Error).message}</Alert>
      ) : (
        <Box sx={{ height: 600, width: '100%' }}>
          <DataGrid<Team>
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

export default Teams; 