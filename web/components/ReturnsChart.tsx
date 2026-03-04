"use client";

import { useState, useEffect } from 'react';


import { LineChart } from '@mui/x-charts/LineChart';
import { loadReturnData } from './server/data';

// types for the CSV rows
interface ReturnRow {
  date: string;
  model: string;
  '2330'?: string;
  '0050'?: string;
}

export default function ReturnsChart({ category, backtest }: { category: string; backtest: string }) {
  const [returnData, setReturnData] = useState<ReturnRow[]>([]);

  useEffect(() => {
    async function fetchData() {
      try {
        const data = await loadReturnData(category, backtest);
        setReturnData(data);
      } catch (error) {
        console.error('Error fetching data:', error);
      }
    }
    fetchData();
  }, [category, backtest]);

  const rows = returnData.filter((r) => r.model && r['2330'] && r['0050']);

  if (rows.length === 0) {
    return <div>No data available for selected backtest.</div>;
  }

  const dates = rows.map((r) => new Date(r.date));
  console.log('Loaded return data:', rows);
  const modelSeries = rows.map((r) => parseFloat(r.model ?? 'NaN'));
  const t2330Series = rows.map((r) => parseFloat(r['2330'] ?? 'NaN'));
  const t0050Series = rows.map((r) => parseFloat(r['0050'] ?? 'NaN'));

  return (
    <LineChart
      xAxis={[
        {
          scaleType: 'time',
          data: dates,
          valueFormatter: (date) => new Intl.DateTimeFormat(undefined, {
            year: 'numeric',
            month: 'short',
            day: 'numeric',
          }).format(date),

        },
      ]}
      series={[
        {
          data: modelSeries,
          label: 'Model',
          showMark: false,
          valueFormatter: (v) => ((v || 0) * 100).toFixed(2) + '%',
        },
        {
          data: t2330Series,
          label: '2330',
          showMark: false,
          valueFormatter: (v) => ((v || 0) * 100).toFixed(2) + '%',
        },
        {
          data: t0050Series,
          label: '0050',
          showMark: false,
          valueFormatter: (v) => ((v || 0) * 100).toFixed(2) + '%',
        },
      ]}
      height={400}
      // theme="light"
      // padding={{ left: 50, right: 20, top: 20, bottom: 40 }}
      // legend={{ position: 'bottom' }}
      yAxis={[{ label: 'Normalized Returns' }]}
    // tooltip={{ formatter: (v) => v.toFixed(4) }}
    />
  );
}

