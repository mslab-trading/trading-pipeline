"use client";

import { useSearchParams } from 'next/navigation';
import { useState, useEffect } from 'react';
import { useRouter } from 'next/navigation';
import { useTheme as useMuiTheme } from '@mui/material/styles';
import { Box, Typography, Select, MenuItem, FormControl, InputLabel, IconButton, Slider } from '@mui/material';
import { DarkMode, LightMode } from '@mui/icons-material';
import { useTheme } from '@/providers/ThemeProvider';
import PortfolioChart from '@/components/PortfolioChart';
import PortfolioPieChart from '@/components/PortfolioPieChart';
import ReturnsChart from '@/components/ReturnsChart';
import TradesTable from '@/components/TradesTable';

const MODELS = ['StockAttentioner', 'BasicModel', 'iTransformer']
const MODEL_MAPPINGS = new Map<string, string>([
  ["StockAttentioner", "StockAttentioner (trading-pipeline)"],
  ["BasicModel", "iTransformer (trading-pipeline)"],
  ["iTransformer", "iTransformer (trading-model)"],
]);


const CATEGORIES = ['Top50', 'Top100', 'Top50_RAM'];
const BACKTESTS = ['allen', 'daily', 'gino'];

export default function Page() {
  const searchParams = useSearchParams();
  const router = useRouter();
  const muiTheme = useMuiTheme();
  const { darkMode, toggleDarkMode } = useTheme();

  const model = searchParams.get('model') || MODELS[0];
  const category = searchParams.get('category') || CATEGORIES[0];
  const backtest = searchParams.get('backtest') || BACKTESTS[0];

  const [selectedModel, setSelectedModel] = useState(model);
  const [selectedCategory, setSelectedCategory] = useState(category);
  const [selectedBacktest, setSelectedBacktest] = useState(backtest);
  const [topN, setTopN] = useState(5);
  const [selectedDate, setSelectedDate] = useState<Date | null>(null);


  useEffect(() => {
    setSelectedModel(model);
  }, [model]);

  useEffect(() => {
    setSelectedCategory(category);
  }, [category]);

  useEffect(() => {
    setSelectedBacktest(backtest);
  }, [backtest]);

  const handleModelChange = (event: any) => {
    const newModel = event.target.value;
    setSelectedModel(newModel);
    router.push(`/?model=${newModel}&category=${selectedCategory}&backtest=${selectedBacktest}`);
  };

  const handleCategoryChange = (event: any) => {
    const newCategory = event.target.value;
    setSelectedCategory(newCategory);
    router.push(`/?model=${selectedModel}&category=${newCategory}&backtest=${selectedBacktest}`);
  };

  const handleBacktestChange = (event: any) => {
    const newBacktest = event.target.value;
    setSelectedBacktest(newBacktest);
    router.push(`/?model=${selectedModel}&category=${selectedCategory}&backtest=${newBacktest}`);
  };

  const handleTopNChange = (event: any) => {
    const newTopN = parseInt(event.target.value, 10);
    setTopN(newTopN);
  };

  console.log('Model from URL', model);
  console.log('Category from URL:', category);
  console.log('Backtest from URL:', backtest);

  if (!MODELS.includes(model) || !CATEGORIES.includes(category) || !BACKTESTS.includes(backtest)) {
    if (!MODELS.includes(model)) {
      return <div>Invalid model: {model} Should be one of ({MODELS.join(', ')})</div>;
    }
    if (!CATEGORIES.includes(category)) {
      return <div>Invalid category: {category} Should be one of ({CATEGORIES.join(', ')})</div>;
    }
    if (!BACKTESTS.includes(backtest)) {
      return <div>Invalid backtest: {backtest} Should be one of ({BACKTESTS.join(', ')})</div>;
    }
  }

  return (
    <Box
      component="main"
      sx={{
        minHeight: '100vh',
        backgroundColor: muiTheme.palette.background.default,
        transition: 'background-color 0.3s ease',
      }}
    >
      <div className="max-w-7xl mx-auto p-6">
        {/* Header */}
        <div className="mb-8 flex justify-between items-start">
          <div>
            <Typography
              variant="h3"
              component="h1"
              sx={{
                fontWeight: 'bold',
                mb: 1,
                color: 'text.primary',
                fontSize: '2.25rem',
              }}
            >
              Backtest Dashboard
            </Typography>
            <Typography
              sx={{
                color: 'text.secondary',
              }}
            >
              Category:{' '}
              <strong style={{ color: 'var(--mui-palette-text-primary)' }}>
                {category}
              </strong>
              {' '}
              Backtest:{' '}
              <strong style={{ color: 'var(--mui-palette-text-primary)' }}>
                {backtest}
              </strong>
            </Typography>
          </div>
          <IconButton
            onClick={toggleDarkMode}
            sx={{
              color: muiTheme.palette.primary.main,
              backgroundColor: muiTheme.palette.background.paper,
              boxShadow: muiTheme.shadows[2],
              '&:hover': {
                backgroundColor: muiTheme.palette.action?.hover,
              },
            }}
          >
            {darkMode ? <LightMode /> : <DarkMode />}
          </IconButton>
        </div>

        {/* Controls Panel */}
        <Box
          sx={{
            mb: 6,
            p: 4,
            backgroundColor: muiTheme.palette.background.paper,
            borderRadius: 2,
            boxShadow: muiTheme.shadows[2],
            border: `1px solid ${muiTheme.palette.divider}`,
            transition: 'all 0.3s ease',
          }}
        >
          <Typography
            variant="h6"
            sx={{
              mb: 3,
              fontWeight: 'bold',
              color: 'text.primary',
            }}
          >
            Controls
          </Typography>
          <Box
            sx={{
              display: 'flex',
              gap: 3,
              flexWrap: 'wrap',
              alignItems: 'flex-end',
            }}
          >
            <FormControl sx={{ minWidth: 220 }}>
              <InputLabel sx={{ color: 'text.secondary' }}>
                Model
              </InputLabel>
              <Select
                value={selectedModel}
                onChange={handleModelChange}
                label="Model"
                size="medium"
                sx={{
                  color: 'text.primary',
                  backgroundColor: muiTheme.palette.background.paper,
                }}
              >
                {MODELS.map((m) => (
                  <MenuItem key={m} value={m}>
                    {MODEL_MAPPINGS.get(m) || m}
                  </MenuItem>
                ))}
              </Select>
            </FormControl>
            <FormControl sx={{ minWidth: 220 }}>
              <InputLabel sx={{ color: 'text.secondary' }}>
                Category
              </InputLabel>
              <Select
                value={selectedCategory}
                onChange={handleCategoryChange}
                label="Category"
                size="medium"
                sx={{
                  color: 'text.primary',
                  backgroundColor: muiTheme.palette.background.paper,
                }}
              >
                {CATEGORIES.map((cat) => (
                  <MenuItem key={cat} value={cat}>
                    {cat}
                  </MenuItem>
                ))}
              </Select>
            </FormControl>
            <FormControl sx={{ minWidth: 220 }}>
              <InputLabel sx={{ color: muiTheme.palette.text.secondary }}>
                Strategy
              </InputLabel>
              <Select
                value={selectedBacktest}
                onChange={handleBacktestChange}
                label="Backtest"
                size="medium"
                sx={{
                  color: 'text.primary',
                  backgroundColor: muiTheme.palette.background.paper,
                }}
              >
                {BACKTESTS.map((bt) => (
                  <MenuItem key={bt} value={bt}>
                    {bt}
                  </MenuItem>
                ))}
              </Select>
            </FormControl>
          </Box>
        </Box>
        <Box
          className="lg:row-span-2"
          sx={{
            backgroundColor: muiTheme.palette.background.paper,
            borderRadius: 2,
            boxShadow: muiTheme.shadows[2],
            border: `1px solid ${muiTheme.palette.divider}`,
            p: 2,
            transition: 'all 0.3s ease',
            height: '100%',
            mb: 6
          }}
        >
          <Typography
            variant="h6"
            sx={{
              mb: 2,
              fontWeight: 'bold',
              color: 'text.primary',
            }}
          >
            Returns Chart
          </Typography>
          <ReturnsChart model={model} category={category} backtest={backtest} />
        </Box>

        {/* Layout: Portfolio Chart and Composition side-by-side on right */}
        <Box
          sx={{
            backgroundColor: muiTheme.palette.background.paper,
            borderRadius: 2,
            boxShadow: muiTheme.shadows[2],
            border: `1px solid ${muiTheme.palette.divider}`,
            p: 2,
            transition: 'all 0.3s ease',
            mb: 6
          }}
        >
          <Typography
            variant="h6"
            sx={{
              mb: 2,
              fontWeight: 'bold',
              color: 'text.primary',
            }}
          >
            Portfolio Chart
          </Typography>
          <Box sx={{ ml: 1, mt: 2, width: 300, mb: 2, borderRadius: 2, padding: `15px 15px 10px 15px`, border: `1px solid ${muiTheme.palette.divider}` }}>
            <InputLabel sx={{ color: muiTheme.palette.text.secondary }}>
              Top N Assets
            </InputLabel>
            <Slider
              value={topN}
              onChange={handleTopNChange}
              valueLabelDisplay="auto"
              min={1}
              max={50}
              marks={[
                { value: 1, label: '1' },
                { value: 25, label: '25' },
                { value: 50, label: '50' },
              ]}
            />
          </Box>
          <div className="grid grid-cols-1 lg:grid-cols-2 gap-6 mb-6">
            <PortfolioChart model={model} category={category} backtest={backtest} topN={topN} setSelectedDate={(d: Date) => setSelectedDate(d)} />
            <Box
              sx={{
                flex: 1,
                maxHeight: '100%',
                overflow: 'auto',
                borderRadius: 4,
                border: `2px solid ${muiTheme.palette.divider}`,
                padding: 2,
              }}
            >
              <div>
                {selectedDate &&
                  <>
                    Portfolio Composition on {' '}
                    {new Intl.DateTimeFormat(undefined, {
                      year: 'numeric',
                      month: 'short',
                      day: 'numeric',
                    }).format(selectedDate)}
                  </>}
                <PortfolioPieChart model={model} category={category} backtest={backtest} topN={topN} selectedDate={selectedDate} setSelectedDate={(d: Date) => setSelectedDate(d)} />
              </div>
            </Box>
          </div>
        </Box>

        {/* Trades Table */}
        <Box
          sx={{
            mb: 6,
            backgroundColor: muiTheme.palette.background.paper,
            borderRadius: 2,
            boxShadow: muiTheme.shadows[2],
            border: `1px solid ${muiTheme.palette.divider}`,
            p: 2,
            overflow: 'auto',
            transition: 'all 0.3s ease',
          }}
        >
          <Typography
            variant="h6"
            sx={{
              mb: 2,
              fontWeight: 'bold',
              color: 'text.primary',
            }}
          >
            Trades
          </Typography>
          <TradesTable model={model} category={category} backtest={backtest} />
        </Box>
      </div>
    </Box>
  );
}
