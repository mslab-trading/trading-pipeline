"use client";

import { loadInfoData } from './server/data';
import { useState, useEffect, use } from 'react';

export default function InfoBox({model, category, backtest}: {model: string, category: string, backtest: string}) {
  const [info, setInfo] = useState<string | null>(null);
  useEffect(() => {
    async function fetchData() {
      try {
        const data = await loadInfoData(model, category, backtest);
        setInfo(data);
      }
      catch (error) {
        console.error('Error fetching data:', error);
      }
    }
    fetchData();
  }, [model, category, backtest]);

  return (
    <>
      <h1>Data from Server File:</h1>
      <p>{info}</p>
    </>
  );

}
