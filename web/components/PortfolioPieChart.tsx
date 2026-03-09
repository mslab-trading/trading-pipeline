"use client";

import { useState, useEffect } from 'react';

import { loadPortfolioData } from './server/data';
import { Slider, Box, Typography } from '@mui/material';
import { PieChart } from '@mui/x-charts';

export default function PortfolioPieChart({ model, category, backtest, topN, selectedDate, setSelectedDate }: { model: string; category: string; backtest: string; topN: number; selectedDate?: Date | null; setSelectedDate: (d: Date) => void}) {
    const [portfolioData, setPortfolioData] = useState<any[]>([]);

    useEffect(() => {
        let active = true;
        async function fetchData() {
            try {
                const data = await loadPortfolioData(model, category, backtest);
                if (active) {
                    setPortfolioData(data);
                    setSelectedDate(new Date(data[data.length - 1].date));
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

    const selectedPortfolio = selectedDate
        ? portfolioData.find(row => new Date(row.date).getTime() === selectedDate.getTime())
        : portfolioData[portfolioData.length - 1];

    const cash = selectedPortfolio ? parseFloat(selectedPortfolio.cash || '0') : 0;
    const entries = Object.entries(selectedPortfolio || {})
        .filter(([key]) => key !== 'date' && key !== 'cash')
        .map(([key, value]) => ({
            id: key,
            value: typeof value === 'string' ? parseFloat(value) || 0 : 0,
            label: key,
        }))
        .filter(entry => entry.value > 0)
        .sort((a, b) => b.value - a.value);
    const otherValue = entries.slice(topN).reduce((acc, cur) => acc + cur.value, 0);
    const simplified_entries = [{ id: 'Cash', value: cash, label: 'Cash' }, ...entries.slice(0, topN), { id: 'Others', value: otherValue, label: 'Others' }].filter(entry => entry.value > 0);
    const totalValue = simplified_entries.reduce((acc, cur) => acc + cur.value, 0);

    return (
        <div suppressHydrationWarning>
            {portfolioData.length === 0 ? (
                <div>No data available for selected backtest.</div>
            ) : (
                <PieChart
                    series={[
                        {
                            data: simplified_entries,
                            valueFormatter: (item: any) => {
                                return ((item.value || 0) / totalValue * 100).toFixed(2) + '%';
                            }
                        },
                    ]}
                    width={250}
                    height={400}
                />
            )}
        </div>
    );
}
