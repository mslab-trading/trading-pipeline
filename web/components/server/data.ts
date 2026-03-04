"use server";

import { promises as fs } from 'fs';
import path from 'path';
import { parse } from 'csv-parse/sync';

async function loadInfoData(category: string, backtest: string): Promise<string | null> {
    const cwd = process.cwd();
    const filePath = path.join(cwd, 'results_backtest', `${category}_${backtest}`, 'info.txt');
    try {
        // Read the file content
        const fileContents = await fs.readFile(filePath, 'utf8');
        return fileContents;
    } catch (error) {
        console.error('Error reading file:', error);
        throw error;
    }
}

async function loadPortfolioData(category: string, backtest: string): Promise<any[]> {
  const cwd = process.cwd();
  const filePath = path.join(cwd, 'results_backtest', `${category}_${backtest}`, 'portfolio_value.csv');

  const fileContents = await fs.readFile(filePath, 'utf8');
  const data = parse(fileContents, {
    columns: true,
    skip_empty_lines: true,
  });

  return data;
}

async function loadTradesData(category: string, backtest: string): Promise<any[]> {
  const cwd = process.cwd();
    const filePath = path.join(cwd, 'results_backtest', `${category}_${backtest}`, 'trades.csv');
    const fileContents = await fs.readFile(filePath, 'utf8');
    const data = parse(fileContents, {
      columns: true,
      skip_empty_lines: true,
    });

    return data;
}

async function loadReturnData(category: string, backtest: string): Promise<any[]> {
  const cwd = process.cwd();
  const filePath = path.join(cwd, 'results_backtest', `${category}_${backtest}`, 'returns.csv');

  const fileContents = await fs.readFile(filePath, 'utf8');
  const parsed: any[] = parse(fileContents, {
    columns: true,
    skip_empty_lines: true,
  }) as any[];

  return parsed;
}

export { loadInfoData, loadPortfolioData, loadTradesData, loadReturnData };