import React, { useState, useEffect, useRef } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import * as d3 from 'd3';
import { GlassCard } from './GlassCard';
import { 
  BarChart3, 
  TrendingUp, 
  Brain, 
  Target, 
  Zap, 
  Clock,
  Activity,
  Eye,
  Map,
  Award
} from 'lucide-react';

const AdvancedAnalyticsLearningDashboard = ({ userId }) => {
  const [analyticsData, setAnalyticsData] = useState(null);
  const [loading, setLoading] = useState(true);
  const [activeTab, setActiveTab] = useState('overview');
  const [timeRange, setTimeRange] = useState('30');
  
  // Refs for D3 visualizations
  const knowledgeGraphRef = useRef(null);
  const heatMapRef = useRef(null);
  const velocityChartRef = useRef(null);
  const retentionCurveRef = useRef(null);

  useEffect(() => {
    if (userId) {
      fetchAnalyticsData();
    }
  }, [userId, timeRange]);

  const fetchAnalyticsData = async () => {
    try {
      setLoading(true);
      const response = await fetch(
        `${process.env.REACT_APP_BACKEND_URL}/api/analytics/${userId}/comprehensive-dashboard`
      );
      
      if (response.ok) {
        const data = await response.json();
        setAnalyticsData(data);
        
        // Trigger visualizations after data load
        setTimeout(() => {
          renderVisualizations(data);
        }, 100);
      }
    } catch (error) {
      console.error('Error fetching analytics data:', error);
    } finally {
      setLoading(false);
    }
  };

  const renderVisualizations = (data) => {
    if (activeTab === 'knowledge-graph') {
      renderKnowledgeGraph(data.knowledge_graph);
    } else if (activeTab === 'competency') {
      renderCompetencyHeatMap(data.competency_heat_map);
    } else if (activeTab === 'velocity') {
      renderVelocityChart(data.learning_velocity);
    } else if (activeTab === 'retention') {
      renderRetentionCurves(data.retention_curves);
    }
  };

  const renderKnowledgeGraph = (graphData) => {
    if (!knowledgeGraphRef.current || !graphData?.nodes) return;

    const container = d3.select(knowledgeGraphRef.current);
    container.selectAll("*").remove();

    const width = 800;
    const height = 600;
    
    const svg = container
      .append("svg")
      .attr("width", width)
      .attr("height", height)
      .attr("viewBox", [0, 0, width, height]);

    // Create force simulation
    const simulation = d3.forceSimulation(graphData.nodes)
      .force("link", d3.forceLink(graphData.edges).id(d => d.id).distance(100))
      .force("charge", d3.forceManyBody().strength(-300))
      .force("center", d3.forceCenter(width / 2, height / 2));

    // Add links
    const link = svg.append("g")
      .selectAll("line")
      .data(graphData.edges)
      .join("line")
      .attr("stroke", "#64748b")
      .attr("stroke-opacity", 0.6)
      .attr("stroke-width", d => Math.sqrt(d.strength) * 2);

    // Add nodes
    const node = svg.append("g")
      .selectAll("circle")
      .data(graphData.nodes)
      .join("circle")
      .attr("r", d => d.size)
      .attr("fill", d => d.color)
      .attr("stroke", "#fff")
      .attr("stroke-width", 2)
      .call(d3.drag()
        .on("start", dragstarted)
        .on("drag", dragged)
        .on("end", dragended));

    // Add labels
    const text = svg.append("g")
      .selectAll("text")
      .data(graphData.nodes)
      .join("text")
      .text(d => d.name)
      .attr("font-size", 12)
      .attr("font-family", "Inter, sans-serif")
      .attr("fill", "#e2e8f0")
      .attr("text-anchor", "middle");

    // Update positions on simulation tick
    simulation.on("tick", () => {
      link
        .attr("x1", d => d.source.x)
        .attr("y1", d => d.source.y)
        .attr("x2", d => d.target.x)
        .attr("y2", d => d.target.y);

      node
        .attr("cx", d => d.x)
        .attr("cy", d => d.y);

      text
        .attr("x", d => d.x)
        .attr("y", d => d.y + 5);
    });

    function dragstarted(event, d) {
      if (!event.active) simulation.alphaTarget(0.3).restart();
      d.fx = d.x;
      d.fy = d.y;
    }

    function dragged(event, d) {
      d.fx = event.x;
      d.fy = event.y;
    }

    function dragended(event, d) {
      if (!event.active) simulation.alphaTarget(0);
      d.fx = null;
      d.fy = null;
    }
  };

  const renderCompetencyHeatMap = (heatMapData) => {
    if (!heatMapRef.current || !heatMapData?.heat_map_data) return;

    const container = d3.select(heatMapRef.current);
    container.selectAll("*").remove();

    const margin = { top: 50, right: 50, bottom: 100, left: 100 };
    const width = 800 - margin.left - margin.right;
    const height = 400 - margin.bottom - margin.top;

    const svg = container
      .append("svg")
      .attr("width", width + margin.left + margin.right)
      .attr("height", height + margin.top + margin.bottom);

    const g = svg.append("g")
      .attr("transform", `translate(${margin.left},${margin.top})`);

    const data = heatMapData.heat_map_data;
    const concepts = heatMapData.concepts;

    // Create scales
    const xScale = d3.scaleBand()
      .domain(data.map(d => d.date))
      .range([0, width])
      .padding(0.1);

    const yScale = d3.scaleBand()
      .domain(concepts.map(c => c.name))
      .range([0, height])
      .padding(0.1);

    const colorScale = d3.scaleSequential(d3.interpolateViridis)
      .domain([0, 1]);

    // Create heat map cells
    data.forEach(day => {
      concepts.forEach(concept => {
        const competency = day.competencies[concept.id] || { score: 0 };
        
        g.append("rect")
          .attr("x", xScale(day.date))
          .attr("y", yScale(concept.name))
          .attr("width", xScale.bandwidth())
          .attr("height", yScale.bandwidth())
          .attr("fill", colorScale(competency.score))
          .attr("stroke", "#1f2937")
          .attr("stroke-width", 1);
      });
    });

    // Add axes
    g.append("g")
      .attr("transform", `translate(0,${height})`)
      .call(d3.axisBottom(xScale))
      .selectAll("text")
      .attr("transform", "rotate(-45)")
      .style("text-anchor", "end")
      .attr("fill", "#e2e8f0");

    g.append("g")
      .call(d3.axisLeft(yScale))
      .selectAll("text")
      .attr("fill", "#e2e8f0");
  };

  const renderVelocityChart = (velocityData) => {
    if (!velocityChartRef.current || !velocityData?.velocity_data) return;

    const container = d3.select(velocityChartRef.current);
    container.selectAll("*").remove();

    const margin = { top: 20, right: 30, bottom: 40, left: 40 };
    const width = 800 - margin.left - margin.right;
    const height = 300 - margin.top - margin.bottom;

    const svg = container
      .append("svg")
      .attr("width", width + margin.left + margin.right)
      .attr("height", height + margin.top + margin.bottom);

    const g = svg.append("g")
      .attr("transform", `translate(${margin.left},${margin.top})`);

    const data = Object.entries(velocityData.velocity_data).map(([concept, data]) => ({
      concept: data.concept_name,
      velocity: data.average_velocity,
      trend: data.velocity_trend
    }));

    const xScale = d3.scaleBand()
      .domain(data.map(d => d.concept))
      .range([0, width])
      .padding(0.1);

    const yScale = d3.scaleLinear()
      .domain(d3.extent(data, d => d.velocity))
      .nice()
      .range([height, 0]);

    // Create bars
    g.selectAll(".bar")
      .data(data)
      .join("rect")
      .attr("class", "bar")
      .attr("x", d => xScale(d.concept))
      .attr("y", d => yScale(Math.max(0, d.velocity)))
      .attr("width", xScale.bandwidth())
      .attr("height", d => Math.abs(yScale(d.velocity) - yScale(0)))
      .attr("fill", d => d.velocity > 0 ? "#10b981" : "#ef4444");

    // Add axes
    g.append("g")
      .attr("transform", `translate(0,${height})`)
      .call(d3.axisBottom(xScale))
      .selectAll("text")
      .attr("transform", "rotate(-45)")
      .style("text-anchor", "end")
      .attr("fill", "#e2e8f0");

    g.append("g")
      .call(d3.axisLeft(yScale))
      .selectAll("text")
      .attr("fill", "#e2e8f0");
  };

  const renderRetentionCurves = (retentionData) => {
    if (!retentionCurveRef.current || !retentionData?.retention_curves) return;

    const container = d3.select(retentionCurveRef.current);
    container.selectAll("*").remove();

    const margin = { top: 20, right: 30, bottom: 40, left: 40 };
    const width = 800 - margin.left - margin.right;
    const height = 300 - margin.top - margin.bottom;

    const svg = container
      .append("svg")
      .attr("width", width + margin.left + margin.right)
      .attr("height", height + margin.top + margin.bottom);

    const g = svg.append("g")
      .attr("transform", `translate(${margin.left},${margin.top})`);

    const curves = Object.entries(retentionData.retention_curves);
    const colorScale = d3.scaleOrdinal(d3.schemeCategory10);

    const xScale = d3.scaleLinear()
      .domain([0, 168]) // 7 days in hours
      .range([0, width]);

    const yScale = d3.scaleLinear()
      .domain([0, 1])
      .range([height, 0]);

    const line = d3.line()
      .x(d => xScale(d.time_gap_hours))
      .y(d => yScale(d.retention_score))
      .curve(d3.curveMonotoneX);

    // Draw retention curves
    curves.forEach(([conceptId, curveData], index) => {
      g.append("path")
        .datum(curveData.retention_points)
        .attr("fill", "none")
        .attr("stroke", colorScale(index))
        .attr("stroke-width", 2)
        .attr("d", line);
    });

    // Add axes
    g.append("g")
      .attr("transform", `translate(0,${height})`)
      .call(d3.axisBottom(xScale))
      .selectAll("text")
      .attr("fill", "#e2e8f0");

    g.append("g")
      .call(d3.axisLeft(yScale))
      .selectAll("text")
      .attr("fill", "#e2e8f0");
  };

  const tabs = [
    { id: 'overview', label: 'Overview', icon: BarChart3 },
    { id: 'knowledge-graph', label: 'Knowledge Graph', icon: Map },
    { id: 'competency', label: 'Competency Heat Map', icon: Activity },
    { id: 'velocity', label: 'Learning Velocity', icon: TrendingUp },
    { id: 'retention', label: 'Retention Curves', icon: Brain },
    { id: 'path-optimization', label: 'Learning Path', icon: Target }
  ];

  if (loading) {
    return (
      <div className="flex items-center justify-center h-96">
        <div className="text-center">
          <div className="animate-spin rounded-full h-16 w-16 border-b-2 border-ai-blue-500 mx-auto mb-4"></div>
          <p className="text-text-secondary">Loading advanced analytics...</p>
        </div>
      </div>
    );
  }

  if (!analyticsData) {
    return (
      <div className="text-center py-12">
        <Brain className="w-16 h-16 text-text-secondary mx-auto mb-4" />
        <p className="text-text-secondary">No analytics data available yet.</p>
        <p className="text-text-tertiary text-sm mt-2">Start learning to see your personalized insights!</p>
      </div>
    );
  }

  return (
    <div className="p-6 space-y-6">
      {/* Header */}
      <div className="flex justify-between items-center">
        <div>
          <h1 className="text-2xl font-bold text-text-primary">Advanced Learning Analytics</h1>
          <p className="text-text-secondary">AI-powered insights into your learning journey</p>
        </div>
        
        <div className="flex items-center space-x-4">
          <select
            value={timeRange}
            onChange={(e) => setTimeRange(e.target.value)}
            className="bg-bg-secondary border border-border-primary rounded-lg px-3 py-2 text-text-primary"
          >
            <option value="7">Last 7 days</option>
            <option value="30">Last 30 days</option>
            <option value="90">Last 90 days</option>
          </select>
        </div>
      </div>

      {/* Summary Cards */}
      {activeTab === 'overview' && (
        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-6 mb-8">
          <GlassCard className="p-4">
            <div className="flex items-center">
              <Award className="w-8 h-8 text-ai-blue-500 mr-3" />
              <div>
                <p className="text-text-secondary text-sm">Mastered Concepts</p>
                <p className="text-2xl font-bold text-text-primary">
                  {analyticsData.summary.mastered_concepts} / {analyticsData.summary.total_concepts}
                </p>
              </div>
            </div>
          </GlassCard>

          <GlassCard className="p-4">
            <div className="flex items-center">
              <Target className="w-8 h-8 text-ai-green-500 mr-3" />
              <div>
                <p className="text-text-secondary text-sm">Overall Competency</p>
                <p className="text-2xl font-bold text-text-primary">
                  {(analyticsData.summary.overall_competency * 100).toFixed(1)}%
                </p>
              </div>
            </div>
          </GlassCard>

          <GlassCard className="p-4">
            <div className="flex items-center">
              <Zap className="w-8 h-8 text-ai-yellow-500 mr-3" />
              <div>
                <p className="text-text-secondary text-sm">Learning Velocity</p>
                <p className="text-2xl font-bold text-text-primary">
                  {analyticsData.summary.learning_velocity.toFixed(2)}
                </p>
              </div>
            </div>
          </GlassCard>

          <GlassCard className="p-4">
            <div className="flex items-center">
              <Clock className="w-8 h-8 text-ai-purple-500 mr-3" />
              <div>
                <p className="text-text-secondary text-sm">Retention Score</p>
                <p className="text-2xl font-bold text-text-primary">
                  {(analyticsData.summary.retention_score * 100).toFixed(1)}%
                </p>
              </div>
            </div>
          </GlassCard>
        </div>
      )}

      {/* Tab Navigation */}
      <div className="flex space-x-2 overflow-x-auto">
        {tabs.map((tab) => {
          const Icon = tab.icon;
          return (
            <button
              key={tab.id}
              onClick={() => {
                setActiveTab(tab.id);
                setTimeout(() => renderVisualizations(analyticsData), 100);
              }}
              className={`flex items-center space-x-2 px-4 py-2 rounded-lg whitespace-nowrap transition-all ${
                activeTab === tab.id
                  ? 'bg-ai-blue-500 text-white'
                  : 'bg-bg-secondary text-text-secondary hover:bg-bg-tertiary'
              }`}
            >
              <Icon className="w-4 h-4" />
              <span>{tab.label}</span>
            </button>
          );
        })}
      </div>

      {/* Content Area */}
      <GlassCard className="p-6">
        <AnimatePresence mode="wait">
          <motion.div
            key={activeTab}
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            exit={{ opacity: 0, y: -20 }}
            transition={{ duration: 0.3 }}
          >
            {activeTab === 'overview' && (
              <div className="space-y-6">
                <h3 className="text-xl font-semibold text-text-primary">Learning Overview</h3>
                
                {/* Next Priority Concepts */}
                <div>
                  <h4 className="text-lg font-medium text-text-primary mb-3">Next Priority Concepts</h4>
                  <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
                    {analyticsData.summary.next_priority_concepts.slice(0, 3).map((conceptId, index) => (
                      <div key={conceptId} className="bg-bg-secondary rounded-lg p-4">
                        <div className="flex items-center justify-between">
                          <h5 className="font-medium text-text-primary">Concept {index + 1}</h5>
                          <span className="text-xs bg-ai-blue-500 text-white px-2 py-1 rounded">
                            Priority
                          </span>
                        </div>
                        <p className="text-text-secondary text-sm mt-2">{conceptId}</p>
                      </div>
                    ))}
                  </div>
                </div>
              </div>
            )}

            {activeTab === 'knowledge-graph' && (
              <div>
                <h3 className="text-xl font-semibold text-text-primary mb-4">Knowledge Graph Mapping</h3>
                <p className="text-text-secondary mb-4">
                  Interactive visualization of concepts and their relationships based on your learning progress.
                </p>
                <div ref={knowledgeGraphRef} className="w-full"></div>
              </div>
            )}

            {activeTab === 'competency' && (
              <div>
                <h3 className="text-xl font-semibold text-text-primary mb-4">Competency Heat Map</h3>
                <p className="text-text-secondary mb-4">
                  Visual representation of your competency levels across different concepts over time.
                </p>
                <div ref={heatMapRef} className="w-full"></div>
              </div>
            )}

            {activeTab === 'velocity' && (
              <div>
                <h3 className="text-xl font-semibold text-text-primary mb-4">Learning Velocity Tracking</h3>
                <p className="text-text-secondary mb-4">
                  Monitor your learning speed and improvement rate across different concepts.
                </p>
                <div ref={velocityChartRef} className="w-full"></div>
              </div>
            )}

            {activeTab === 'retention' && (
              <div>
                <h3 className="text-xl font-semibold text-text-primary mb-4">Retention Curves</h3>
                <p className="text-text-secondary mb-4">
                  Analyze how well you retain knowledge over time using forgetting curve models.
                </p>
                <div ref={retentionCurveRef} className="w-full"></div>
              </div>
            )}

            {activeTab === 'path-optimization' && (
              <div>
                <h3 className="text-xl font-semibold text-text-primary mb-4">AI-Optimized Learning Path</h3>
                <p className="text-text-secondary mb-4">
                  Personalized learning sequence optimized for your goals, competencies, and learning style.
                </p>
                
                {analyticsData.learning_path_optimization?.optimal_path && (
                  <div className="space-y-4">
                    {analyticsData.learning_path_optimization.optimal_path.slice(0, 10).map((item, index) => (
                      <motion.div
                        key={item.concept_id}
                        initial={{ opacity: 0, x: -20 }}
                        animate={{ opacity: 1, x: 0 }}
                        transition={{ delay: index * 0.1 }}
                        className="bg-bg-secondary rounded-lg p-4 border-l-4 border-ai-blue-500"
                      >
                        <div className="flex justify-between items-start">
                          <div className="flex-1">
                            <h5 className="font-semibold text-text-primary">
                              {item.position}. {item.concept_name}
                            </h5>
                            <p className="text-text-secondary text-sm mt-1">{item.description}</p>
                            <div className="flex items-center space-x-4 mt-3">
                              <span className="text-xs bg-bg-tertiary px-2 py-1 rounded">
                                {item.category}
                              </span>
                              <span className="text-xs text-text-tertiary">
                                Est. {item.estimated_time_hours.toFixed(1)}h
                              </span>
                              <span className="text-xs text-text-tertiary">
                                Readiness: {(item.readiness_score * 100).toFixed(0)}%
                              </span>
                            </div>
                          </div>
                          <div className="text-right">
                            <div className="w-12 h-12 rounded-full bg-ai-blue-500/20 flex items-center justify-center">
                              <span className="text-ai-blue-400 text-sm font-bold">{item.position}</span>
                            </div>
                          </div>
                        </div>
                      </motion.div>
                    ))}
                  </div>
                )}
              </div>
            )}
          </motion.div>
        </AnimatePresence>
      </GlassCard>
    </div>
  );
};

export default AdvancedAnalyticsLearningDashboard;