"use client";

import { useState, useEffect } from 'react';
import { useRef } from 'react';

import { LineChart } from '@mui/x-charts/LineChart';
import { loadPortfolioData } from './server/data';
import { setDefaultAutoSelectFamily } from 'net';

export default function PortfolioChart({ category, backtest, topN = 5, setSelectedDate }: { category: string; backtest: string; topN?: number; setSelectedDate: (d: Date) => void }) {
  const [portfolioData, setPortfolioData] = useState<any[]>([]);
  const dates = portfolioData.map((row) => new Date(row.date));

  useEffect(() => {
    async function fetchData() {
      try {
        const data = await loadPortfolioData(category, backtest);
        setPortfolioData(data);
      } catch (error) {
        console.error('Error fetching data:', error);
      }
    }
    fetchData();
  }, [category, backtest]);

  console.log(portfolioData);

  const totals = portfolioData.reduce((acc: Record<string, number>, row) => {
    for (const key in row) {
      if (key !== 'date') {
        acc[key] = (acc[key] || 0) + parseFloat(row[key] || '0');
      }
    }
    return acc;
  }, {} as Record<string, number>); // id -> total value

  const sortedStocks = Object.entries(totals)
    .filter(([key]) => key !== 'cash')
    .sort(([, val1], [, val2]) => val2 - val1)
    .map(([id,]) => id);
  const topStocks = sortedStocks.slice(0, topN);
  const otherStocks = sortedStocks.slice(topN);

  const othersValues = portfolioData.map(row => {
    let sum = 0;
    for (const key in row) {
      if (otherStocks.includes(key)) {
        sum += parseFloat(row[key] || '0');
      }
    }
    return sum;
  });

  const entries = (otherStocks.length > 0 ? ['cash', ...topStocks, 'others'] : ['cash', ...topStocks,]).map((key) => ({
    data: key === 'others' ? othersValues : portfolioData.map(row => parseFloat(row[key] || '0')),
    label: key,
    area: true,
    stack: 'total',
    showMark: false,
  }));


  return (
    portfolioData.length === 0 ? (
      <div>No data available for selected backtest.</div>
    ) : (
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
        onAxisClick={(_, data) => {
          setSelectedDate(data?.axisValue as Date);
        }}
        series={entries}
        height={400}
        // theme="light"
        // padding={{ left: 50, right: 20, top: 20, bottom: 40 }}
        // legend={{ position: 'bottom' }}
        yAxis={[{ label: 'Normalized Returns' }]}
      // tooltip={{ formatter: (v) => v.toFixed(4) }}
      />
    ));
}
