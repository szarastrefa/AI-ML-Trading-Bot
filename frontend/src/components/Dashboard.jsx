import React, { useState, useEffect } from 'react';
import { Card, Row, Col, Statistic, Button, Select, message } from 'antd';
import { Line } from '@ant-design/plots';
import { 
  DollarCircleOutlined, 
  TrophyOutlined, 
  WarningOutlined,
  LineChartOutlined
} from '@ant-design/icons';
import axios from 'axios';

const { Option } = Select;

const Dashboard = () => {
  const [loading, setLoading] = useState(false);
  const [accountInfo, setAccountInfo] = useState(null);
  const [pnlData, setPnlData] = useState([]);
  const [selectedPeriod, setSelectedPeriod] = useState('1w');

  useEffect(() => {
    fetchDashboardData();
    const interval = setInterval(fetchDashboardData, 30000);
    return () => clearInterval(interval);
  }, []);

  const fetchDashboardData = async () => {
    setLoading(true);
    try {
      // Mock data for demo
      setAccountInfo({
        balance: 10000,
        equity: 10250,
        free_margin: 8500,
        margin_level: 150,
        currency: 'USD'
      });

      setPnlData([
        { date: '2024-01-01', pnl: 100 },
        { date: '2024-01-02', pnl: 250 },
        { date: '2024-01-03', pnl: 180 },
        { date: '2024-01-04', pnl: 400 },
        { date: '2024-01-05', pnl: 320 }
      ]);

    } catch (error) {
      message.error('Failed to fetch dashboard data');
    } finally {
      setLoading(false);
    }
  };

  const pnlConfig = {
    data: pnlData,
    xField: 'date',
    yField: 'pnl',
    smooth: true,
    color: '#52c41a',
  };

  return (
    <div className="p-6 space-y-6">
      <div className="flex justify-between items-center mb-6">
        <h1 className="text-2xl font-bold text-gray-900">Trading Dashboard</h1>
        <div className="flex space-x-4">
          <Select
            value={selectedPeriod}
            onChange={setSelectedPeriod}
            className="w-32"
          >
            <Option value="1d">1 Day</Option>
            <Option value="1w">1 Week</Option>
            <Option value="1m">1 Month</Option>
            <Option value="3m">3 Months</Option>
            <Option value="1y">1 Year</Option>
            <Option value="all">All Time</Option>
          </Select>
          <Button onClick={fetchDashboardData} loading={loading}>
            Refresh
          </Button>
        </div>
      </div>

      {/* Account Overview */}
      <Row gutter={[16, 16]}>
        <Col xs={24} sm={12} lg={6}>
          <Card>
            <Statistic
              title="Account Balance"
              value={accountInfo?.balance || 0}
              precision={2}
              valueStyle={{ color: '#3f8600' }}
              prefix={<DollarCircleOutlined />}
              suffix={accountInfo?.currency || 'USD'}
            />
          </Card>
        </Col>
        <Col xs={24} sm={12} lg={6}>
          <Card>
            <Statistic
              title="Equity"
              value={accountInfo?.equity || 0}
              precision={2}
              valueStyle={{ color: '#1890ff' }}
              prefix={<TrophyOutlined />}
              suffix={accountInfo?.currency || 'USD'}
            />
          </Card>
        </Col>
        <Col xs={24} sm={12} lg={6}>
          <Card>
            <Statistic
              title="Free Margin"
              value={accountInfo?.free_margin || 0}
              precision={2}
              valueStyle={{ color: '#722ed1' }}
              prefix={<LineChartOutlined />}
              suffix={accountInfo?.currency || 'USD'}
            />
          </Card>
        </Col>
        <Col xs={24} sm={12} lg={6}>
          <Card>
            <Statistic
              title="Margin Level"
              value={accountInfo?.margin_level || 0}
              precision={2}
              valueStyle={{ 
                color: (accountInfo?.margin_level || 0) > 200 ? '#3f8600' : '#cf1322' 
              }}
              prefix={<WarningOutlined />}
              suffix="%"
            />
          </Card>
        </Col>
      </Row>

      {/* P&L Chart */}
      <Card title="Profit & Loss Chart" className="w-full">
        <Line {...pnlConfig} height={300} />
      </Card>
    </div>
  );
};

export default Dashboard;