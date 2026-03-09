"use client"
import { useState, useEffect } from "react";
import Table from '@mui/material/Table';
import TableBody from '@mui/material/TableBody';
import TableCell from '@mui/material/TableCell';
import TableContainer from '@mui/material/TableContainer';
import TableHead from '@mui/material/TableHead';
import TableRow from '@mui/material/TableRow';
import Paper from '@mui/material/Paper';
import { DataGrid, GridColDef } from '@mui/x-data-grid';
import { loadTradesData } from "./server/data"


export default function TradesTable({ model, category, backtest }: { model: string, category: string, backtest: string }) {
    const [trades, setTrades] = useState<any[]>([]);
    const [mounted, setMounted] = useState(false);

    useEffect(() => {
        let active = true;
        async function fetchData() {
            try {
                const data = await loadTradesData(model, category, backtest);
                if (active) {
                    setTrades(data);
                }
            } catch (error) {
                if (active) {
                    console.error('Error fetching data:', error);
                }
            }
        }
        fetchData();
        return () => {
            active = false;
        };
    }, [model, category, backtest]);

    useEffect(() => {
        setMounted(true);
        return () => {
            setMounted(false);
        };
    }, []);


    const columns: GridColDef<(typeof trades)[number]>[] = Object.keys(trades[0] || {}).map((key) => ({
        field: key,
        headerName: key,
        width: 150,
    }));

    return (
        <div>
            <h2>Trades Table</h2>
            <p>This is where the trades table will go.</p>
            <Paper sx={{ height: 400, width: '100%' }}>
                {mounted && (
                  <DataGrid
                      rows={trades}
                      columns={columns}
                      initialState={{ pagination: { paginationModel: {
                pageSize: 5,
              }, } }}
                      pageSizeOptions={[5, 10]}
                      checkboxSelection
                      sx={{ border: 0 }}
                  />
                )}
            </Paper>
        </div>
    );
}