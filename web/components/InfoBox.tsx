"use client";

import { loadInfoData } from './server/data';
import { useState, useEffect, use } from 'react';

export default function InfoBox({category, backtest}: {category: string, backtest: string}) {
  const [info, setInfo] = useState<string | null>(null);
  useEffect(() => {
    async function fetchData() {
      try {
        const data = await loadInfoData(category, backtest);
        setInfo(data);
      }
      catch (error) {
        console.error('Error fetching data:', error);
      }
    }
    fetchData();
  }, [category, backtest]);

  return (
    <>
      <h1>Data from Server File:</h1>
      <p>{info}</p>
    </>
  );

}
