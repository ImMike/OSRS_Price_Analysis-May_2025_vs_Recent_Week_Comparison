"""
Visualizer for RuneScape Price Tracker
Creates interactive charts and tables for price decline data
"""

import pandas as pd
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import json
from datetime import datetime
from typing import Dict, List, Optional
import logging
from tabulate import tabulate
from rich.console import Console
from rich.table import Table
from rich.text import Text
import os

from config.settings import TOP_DECLINING_ITEMS, CHART_WIDTH, CHART_HEIGHT, EXPORTS_PATH

logger = logging.getLogger(__name__)


class PriceVisualizer:
    """Creates visualizations for price decline analysis"""
    
    def __init__(self):
        self.console = Console()
        
        # Set up matplotlib style
        plt.style.use('default')
        plt.rcParams['figure.figsize'] = (12, 8)
        plt.rcParams['font.size'] = 10
    
    def create_decline_bar_chart(self, results_df: pd.DataFrame, top_n: int = TOP_DECLINING_ITEMS) -> go.Figure:
        """
        Create interactive bar chart showing top declining items
        
        Args:
            results_df: DataFrame with analysis results
            top_n: Number of top items to show
        
        Returns:
            Plotly figure object
        """
        if results_df.empty:
            logger.warning("No data available for chart creation")
            return go.Figure()
        
        # Get top declining items
        top_items = results_df.head(top_n).copy()
        
        # Create color scale based on decline percentage
        colors = px.colors.sequential.Reds_r
        
        fig = go.Figure()
        
        # Add bar chart
        fig.add_trace(go.Bar(
            x=top_items['decline_percentage'],
            y=top_items['item_name'],
            orientation='h',
            marker=dict(
                color=top_items['decline_percentage'],
                colorscale='Reds',
                showscale=True,
                colorbar=dict(title="Decline %")
            ),
            text=[f"{p:.1f}%" for p in top_items['decline_percentage']],
            textposition='outside',
            hovertemplate=(
                "<b>%{y}</b><br>" +
                "Decline: %{x:.1f}%<br>" +
                "Current Price: %{customdata[0]:,} gp<br>" +
                "Historical Price: %{customdata[1]:,} gp<br>" +
                "Amount Lost: %{customdata[2]:,} gp<br>" +
                "Method: %{customdata[3]}<br>" +
                "<extra></extra>"
            ),
            customdata=top_items[['current_price', 'historical_price', 'decline_amount', 'calculation_method']].values
        ))
        
        # Update layout
        fig.update_layout(
            title={
                'text': f"Top {top_n} OSRS Items by Price Decline",
                'x': 0.5,
                'xanchor': 'center',
                'font': {'size': 20}
            },
            xaxis_title="Price Decline Percentage (%)",
            yaxis_title="Items",
            width=CHART_WIDTH,
            height=max(CHART_HEIGHT, top_n * 25 + 200),
            margin=dict(l=200, r=100, t=80, b=80),
            showlegend=False,
            hovermode='closest'
        )
        
        # Reverse y-axis to show highest decline at top
        fig.update_yaxes(autorange="reversed")
        
        return fig
    
    def create_decline_distribution(self, results_df: pd.DataFrame) -> go.Figure:
        """
        Create histogram showing distribution of price declines
        
        Args:
            results_df: DataFrame with analysis results
        
        Returns:
            Plotly figure object
        """
        if results_df.empty:
            return go.Figure()
        
        fig = go.Figure()
        
        # Create histogram
        fig.add_trace(go.Histogram(
            x=results_df['decline_percentage'],
            nbinsx=50,
            marker=dict(
                color='rgba(255, 100, 100, 0.7)',
                line=dict(color='rgba(255, 100, 100, 1.0)', width=1)
            ),
            hovertemplate=(
                "Decline Range: %{x}%<br>" +
                "Number of Items: %{y}<br>" +
                "<extra></extra>"
            )
        ))
        
        # Add vertical line for mean
        mean_decline = results_df['decline_percentage'].mean()
        fig.add_vline(
            x=mean_decline,
            line_dash="dash",
            line_color="red",
            annotation_text=f"Mean: {mean_decline:.1f}%"
        )
        
        fig.update_layout(
            title={
                'text': "Distribution of Price Declines",
                'x': 0.5,
                'xanchor': 'center'
            },
            xaxis_title="Price Decline Percentage (%)",
            yaxis_title="Number of Items",
            width=CHART_WIDTH,
            height=600
        )
        
        return fig
    
    def create_method_comparison(self, results_df: pd.DataFrame) -> go.Figure:
        """
        Create chart comparing calculation methods used
        
        Args:
            results_df: DataFrame with analysis results
        
        Returns:
            Plotly figure object
        """
        if results_df.empty:
            return go.Figure()
        
        method_counts = results_df['calculation_method'].value_counts()
        
        fig = go.Figure()
        
        fig.add_trace(go.Pie(
            labels=method_counts.index,
            values=method_counts.values,
            hole=0.3,
            hovertemplate=(
                "<b>%{label}</b><br>" +
                "Items: %{value}<br>" +
                "Percentage: %{percent}<br>" +
                "<extra></extra>"
            )
        ))
        
        fig.update_layout(
            title={
                'text': "Calculation Methods Used",
                'x': 0.5,
                'xanchor': 'center'
            },
            width=600,
            height=500
        )
        
        return fig
    
    def create_summary_dashboard(self, results_df: pd.DataFrame, summary_stats: Dict) -> go.Figure:
        """
        Create comprehensive dashboard with multiple visualizations
        
        Args:
            results_df: DataFrame with analysis results
            summary_stats: Dictionary with summary statistics
        
        Returns:
            Plotly figure with subplots
        """
        if results_df.empty:
            return go.Figure()
        
        # Create subplots
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=(
                f"Top {min(20, len(results_df))} Declining Items",
                "Price Decline Distribution", 
                "Calculation Methods",
                "Summary Statistics"
            ),
            specs=[
                [{"type": "bar"}, {"type": "histogram"}],
                [{"type": "pie"}, {"type": "table"}]
            ],
            vertical_spacing=0.12,
            horizontal_spacing=0.1
        )
        
        # Top declining items (horizontal bar)
        top_items = results_df.head(20)
        fig.add_trace(
            go.Bar(
                x=top_items['decline_percentage'],
                y=top_items['item_name'],
                orientation='h',
                marker_color='red',
                name="Decline %"
            ),
            row=1, col=1
        )
        
        # Distribution histogram
        fig.add_trace(
            go.Histogram(
                x=results_df['decline_percentage'],
                nbinsx=30,
                marker_color='orange',
                name="Distribution"
            ),
            row=1, col=2
        )
        
        # Methods pie chart
        method_counts = results_df['calculation_method'].value_counts()
        fig.add_trace(
            go.Pie(
                labels=method_counts.index,
                values=method_counts.values,
                name="Methods"
            ),
            row=2, col=1
        )
        
        # Summary table
        if summary_stats:
            summary_data = [
                ["Total Items", f"{summary_stats.get('total_items', 0):,}"],
                ["Items Declining", f"{summary_stats.get('items_with_decline', 0):,}"],
                ["Items Increasing", f"{summary_stats.get('items_with_increase', 0):,}"],
                ["Avg Decline %", f"{summary_stats.get('average_decline_percentage', 0):.1f}%"],
                ["Max Decline %", f"{summary_stats.get('max_decline_percentage', 0):.1f}%"],
                ["Total Value Lost", f"{summary_stats.get('total_value_lost', 0):,.0f} gp"]
            ]
            
            fig.add_trace(
                go.Table(
                    header=dict(values=["Metric", "Value"], fill_color="lightgray"),
                    cells=dict(values=list(zip(*summary_data)), fill_color="white")
                ),
                row=2, col=2
            )
        
        fig.update_layout(
            title={
                'text': "OSRS Price Decline Analysis Dashboard",
                'x': 0.5,
                'xanchor': 'center',
                'font': {'size': 24}
            },
            width=1400,
            height=1000,
            showlegend=False
        )
        
        return fig
    
    def print_console_table(self, results_df: pd.DataFrame, top_n: int = 20) -> None:
        """
        Print formatted table to console using Rich
        
        Args:
            results_df: DataFrame with analysis results
            top_n: Number of top items to show
        """
        if results_df.empty:
            self.console.print("[red]No data available to display[/red]")
            return
        
        # Create Rich table
        table = Table(title=f"Top {top_n} OSRS Items by Price Decline")
        
        table.add_column("Rank", justify="right", style="cyan")
        table.add_column("Item Name", style="magenta")
        table.add_column("Current Price", justify="right", style="green")
        table.add_column("Historical Price", justify="right", style="blue")
        table.add_column("Decline %", justify="right", style="red")
        table.add_column("Amount Lost", justify="right", style="yellow")
        table.add_column("Method", justify="center", style="white")
        
        # Add rows
        for idx, row in results_df.head(top_n).iterrows():
            table.add_row(
                str(idx + 1),
                row['item_name'][:30],  # Truncate long names
                f"{row['current_price']:,} gp",
                f"{row['historical_price']:,} gp",
                f"{row['decline_percentage']:.1f}%",
                f"{row['decline_amount']:,} gp",
                row['calculation_method']
            )
        
        self.console.print(table)
    
    def save_results_csv(self, results_df: pd.DataFrame, filename: Optional[str] = None) -> str:
        """
        Save results to CSV file
        
        Args:
            results_df: DataFrame with analysis results
            filename: Optional custom filename
        
        Returns:
            Path to saved file
        """
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"osrs_price_decline_{timestamp}.csv"
        
        filepath = f"{EXPORTS_PATH}/{filename}"
        results_df.to_csv(filepath, index=False)
        
        logger.info(f"Results saved to {filepath}")
        return filepath
    
    def save_chart_html(self, fig: go.Figure, filename: Optional[str] = None) -> str:
        """
        Save interactive chart as HTML file
        
        Args:
            fig: Plotly figure object
            filename: Optional custom filename
        
        Returns:
            Path to saved file
        """
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"osrs_price_chart_{timestamp}.html"
        
        filepath = f"{EXPORTS_PATH}/{filename}"
        fig.write_html(filepath)
        
        logger.info(f"Chart saved to {filepath}")
        return filepath
    
    def create_interactive_data_table(self, results_df: pd.DataFrame) -> go.Figure:
        """
        Create interactive data table with filtering and sorting capabilities
        
        Args:
            results_df: DataFrame with analysis results
        
        Returns:
            Plotly figure with interactive table
        """
        if results_df.empty:
            return go.Figure()
        
        # Prepare data for display
        display_df = results_df.copy()
        
        # Format numeric columns for better display (with NaN handling)
        def safe_format(formatter):
            def wrapper(x):
                if pd.isna(x):
                    return "N/A"
                return formatter(x)
            return wrapper
        
        numeric_columns = {
            'current_price': safe_format(lambda x: f"{x:,.0f} gp"),
            'historical_price': safe_format(lambda x: f"{x:,.0f} gp"),
            'decline_amount': safe_format(lambda x: f"{x:,.0f} gp"),
            'decline_percentage': safe_format(lambda x: f"{x:.1f}%"),
            'net_profit_percentage': safe_format(lambda x: f"{x:.1f}%"),
            'liquidity_score': safe_format(lambda x: f"{x:.0f}"),
            'stability_score': safe_format(lambda x: f"{x:.0f}"),
            'quality_score': safe_format(lambda x: f"{x:.0f}"),
            'opportunity_score': safe_format(lambda x: f"{x:.0f}"),
            'adjusted_opportunity_score': safe_format(lambda x: f"{x:.0f}"),
            'avg_daily_volume': safe_format(lambda x: f"{x:.0f}"),
            'bid_ask_spread_percentage': safe_format(lambda x: f"{x:.1f}%"),
            'total_slippage_percentage': safe_format(lambda x: f"{x:.1f}%"),
            'price_volatility': safe_format(lambda x: f"{x:.1f}%"),
            'outlier_ratio': safe_format(lambda x: f"{x:.1%}"),
            'trend_strength': safe_format(lambda x: f"{x:.0f}"),
            'support_level': safe_format(lambda x: f"{x:,.0f} gp"),
            'resistance_level': safe_format(lambda x: f"{x:,.0f} gp")
        }
        
        for col, formatter in numeric_columns.items():
            if col in display_df.columns:
                display_df[f"{col}_formatted"] = display_df[col].apply(formatter)
        
        # Select and order columns for display
        display_columns = [
            'item_name', 'current_price_formatted', 'historical_price_formatted',
            'decline_percentage_formatted', 'net_profit_percentage_formatted',
            'enhanced_recommendation', 'liquidity_score_formatted', 
            'stability_score_formatted', 'quality_score_formatted',
            'is_stable_market', 'is_genuine_decline', 'trend_type',
            'execution_feasibility', 'avg_daily_volume_formatted',
            'bid_ask_spread_percentage_formatted', 'total_slippage_percentage_formatted'
        ]
        
        # Filter to only include columns that exist
        available_columns = [col for col in display_columns if col in display_df.columns]
        table_data = display_df[available_columns]
        
        # Create column headers with better names
        column_names = {
            'item_name': 'Item Name',
            'current_price_formatted': 'Current Price',
            'historical_price_formatted': 'Historical Price',
            'decline_percentage_formatted': 'Decline %',
            'net_profit_percentage_formatted': 'Net Profit %',
            'enhanced_recommendation': 'Recommendation',
            'liquidity_score_formatted': 'Liquidity Score',
            'stability_score_formatted': 'Stability Score',
            'quality_score_formatted': 'Quality Score',
            'is_stable_market': 'Stable Market',
            'is_genuine_decline': 'Genuine Decline',
            'trend_type': 'Trend Type',
            'execution_feasibility': 'Execution',
            'avg_daily_volume_formatted': 'Daily Volume',
            'bid_ask_spread_percentage_formatted': 'Spread %',
            'total_slippage_percentage_formatted': 'Slippage %'
        }
        
        headers = [column_names.get(col, col.replace('_', ' ').title()) for col in available_columns]
        
        # Create color coding for recommendations
        def get_recommendation_color(rec):
            colors = {
                'strong_buy': '#00ff00',
                'buy': '#90EE90', 
                'consider': '#FFD700',
                'avoid': '#FF6B6B'
            }
            return colors.get(rec, '#FFFFFF')
        
        # Prepare cell colors
        cell_colors = []
        for col in available_columns:
            if col == 'enhanced_recommendation':
                colors = [get_recommendation_color(val) for val in table_data[col]]
                cell_colors.append(colors)
            else:
                cell_colors.append(['white'] * len(table_data))
        
        fig = go.Figure(data=[go.Table(
            header=dict(
                values=headers,
                fill_color='lightblue',
                align='left',
                font=dict(size=12, color='black'),
                height=40
            ),
            cells=dict(
                values=[table_data[col].tolist() for col in available_columns],
                fill_color=cell_colors,
                align='left',
                font=dict(size=11),
                height=30
            )
        )])
        
        fig.update_layout(
            title={
                'text': "OSRS Price Analysis - Interactive Data Table (Scroll to see all columns)",
                'x': 0.5,
                'xanchor': 'center',
                'font': {'size': 20}
            },
            width=1600,
            height=800
        )
        
        return fig
    
    def create_sortable_html_table(self, results_df: pd.DataFrame) -> str:
        """
        Create a sortable HTML table with DataTables for better data interaction
        
        Args:
            results_df: DataFrame with analysis results
        
        Returns:
            Path to saved HTML table file
        """
        if results_df.empty:
            return ""
        
        # Prepare data for HTML table
        display_df = results_df.copy()
        
        # Select key columns for the sortable table
        key_columns = [
            'item_name', 'may_2025_price', 'recent_price', 'current_price', 'decline_percentage',
            'net_profit_percentage', 'calculation_method', 'enhanced_recommendation', 'liquidity_score',
            'stability_score', 'quality_score', 'adjusted_opportunity_score',
            'is_stable_market', 'is_genuine_decline', 'trend_type', 'execution_feasibility',
            'avg_daily_volume', 'bid_ask_spread_percentage', 'total_slippage_percentage'
        ]
        
        # Filter to only include columns that exist
        available_columns = [col for col in key_columns if col in display_df.columns]
        table_df = display_df[available_columns].copy()
        
        # Format numeric columns for display
        if 'may_2025_price' in table_df.columns:
            table_df['may_2025_price'] = table_df['may_2025_price'].apply(lambda x: f"{x:,.0f}" if pd.notna(x) else "N/A")
        if 'recent_price' in table_df.columns:
            table_df['recent_price'] = table_df['recent_price'].apply(lambda x: f"{x:,.0f}" if pd.notna(x) else "N/A")
        table_df['current_price'] = table_df['current_price'].apply(lambda x: f"{x:,.0f}" if pd.notna(x) else "N/A")
        table_df['decline_percentage'] = table_df['decline_percentage'].apply(lambda x: f"{x:.1f}%" if pd.notna(x) else "N/A")
        table_df['net_profit_percentage'] = table_df['net_profit_percentage'].apply(lambda x: f"{x:.1f}%" if pd.notna(x) else "N/A")
        table_df['liquidity_score'] = table_df['liquidity_score'].apply(lambda x: f"{x:.0f}" if pd.notna(x) else "N/A")
        table_df['stability_score'] = table_df['stability_score'].apply(lambda x: f"{x:.0f}" if pd.notna(x) else "N/A")
        table_df['quality_score'] = table_df['quality_score'].apply(lambda x: f"{x:.0f}" if pd.notna(x) else "N/A")
        table_df['adjusted_opportunity_score'] = table_df['adjusted_opportunity_score'].apply(lambda x: f"{x:.0f}" if pd.notna(x) else "N/A")
        table_df['avg_daily_volume'] = table_df['avg_daily_volume'].apply(lambda x: f"{x:.0f}" if pd.notna(x) else "N/A")
        table_df['bid_ask_spread_percentage'] = table_df['bid_ask_spread_percentage'].apply(lambda x: f"{x:.1f}%" if pd.notna(x) else "N/A")
        table_df['total_slippage_percentage'] = table_df['total_slippage_percentage'].apply(lambda x: f"{x:.1f}%" if pd.notna(x) else "N/A")
        
        # Create better column names
        column_names = {
            'item_name': 'Item Name',
            'may_2025_price': 'May 2025 Avg (gp)',
            'recent_price': 'Recent Week Avg (gp)',
            'current_price': 'Current Price (gp)',
            'decline_percentage': 'Decline %',
            'net_profit_percentage': 'Net Profit %',
            'calculation_method': 'Method',
            'enhanced_recommendation': 'Recommendation',
            'liquidity_score': 'Liquidity Score',
            'stability_score': 'Stability Score',
            'quality_score': 'Quality Score',
            'adjusted_opportunity_score': 'Opportunity Score',
            'is_stable_market': 'Stable Market',
            'is_genuine_decline': 'Genuine Decline',
            'trend_type': 'Trend Type',
            'execution_feasibility': 'Execution',
            'avg_daily_volume': 'Daily Volume',
            'bid_ask_spread_percentage': 'Spread %',
            'total_slippage_percentage': 'Slippage %'
        }
        
        table_df.columns = [column_names.get(col, col) for col in table_df.columns]
        
        # Generate HTML with DataTables
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        
        html_content = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <title>OSRS Price Analysis - Sortable Data Table</title>
            <link rel="stylesheet" type="text/css" href="https://cdn.datatables.net/1.13.6/css/jquery.dataTables.min.css">
            <script type="text/javascript" charset="utf8" src="https://code.jquery.com/jquery-3.6.0.min.js"></script>
            <script type="text/javascript" charset="utf8" src="https://cdn.datatables.net/1.13.6/js/jquery.dataTables.min.js"></script>
            <style>
                body {{
                    font-family: Arial, sans-serif;
                    margin: 20px;
                    background-color: #f5f5f5;
                }}
                .header {{
                    text-align: center;
                    background-color: #2c3e50;
                    color: white;
                    padding: 20px;
                    border-radius: 10px;
                    margin-bottom: 20px;
                }}
                .table-container {{
                    background: white;
                    padding: 20px;
                    border-radius: 10px;
                    box-shadow: 0 2px 8px rgba(0,0,0,0.1);
                }}
                .recommendation-strong_buy {{ background-color: #d4edda !important; }}
                .recommendation-buy {{ background-color: #d1ecf1 !important; }}
                .recommendation-consider {{ background-color: #fff3cd !important; }}
                .recommendation-avoid {{ background-color: #f8d7da !important; }}
                .dataTables_wrapper .dataTables_length select {{
                    padding: 5px;
                }}
                .dataTables_wrapper .dataTables_filter input {{
                    padding: 5px;
                    margin-left: 10px;
                }}
                table.dataTable thead th {{
                    background-color: #3498db;
                    color: white;
                    cursor: pointer;
                }}
                table.dataTable thead th:hover {{
                    background-color: #2980b9;
                }}
                .btn {{
                    display: inline-block;
                    padding: 4px 8px;
                    margin: 2px;
                    font-size: 12px;
                    font-weight: bold;
                    text-align: center;
                    text-decoration: none;
                    border: 1px solid transparent;
                    border-radius: 4px;
                    cursor: pointer;
                    transition: all 0.2s;
                }}
                .btn-primary {{ background-color: #007bff; color: white; border-color: #007bff; }}
                .btn-success {{ background-color: #28a745; color: white; border-color: #28a745; }}
                .btn-info {{ background-color: #17a2b8; color: white; border-color: #17a2b8; }}
                .btn-warning {{ background-color: #ffc107; color: black; border-color: #ffc107; }}
                .btn-secondary {{ background-color: #6c757d; color: white; border-color: #6c757d; }}
                .btn:hover {{ opacity: 0.8; transform: translateY(-1px); }}
            </style>
        </head>
        <body>
            <div class="header">
                <h1>🏰 OSRS Price Analysis - May 2025 vs Recent Week Comparison</h1>
                <p><strong>Methodology:</strong> Compares May 2025 average prices vs Last 7 days average prices</p>
                <p>Click column headers to sort • Use search box to filter • Change "Show entries" to see more rows</p>
                <p><small>Generated: {timestamp}</small></p>
            </div>
            
            <div class="table-container">
                <table id="priceTable" class="display" style="width:100%">
                    <thead>
                        <tr>
                            {''.join(f'<th>{col}</th>' for col in table_df.columns)}
                        </tr>
                    </thead>
                    <tbody>
        """
        
        # Add table rows with proper data attributes for sorting
        numeric_columns = ['May 2025 Avg (gp)', 'Recent Week Avg (gp)', 'Current Price (gp)', 'Decline %', 
                          'Net Profit %', 'Liquidity Score', 'Stability Score', 'Quality Score', 
                          'Opportunity Score', 'Daily Volume', 'Spread %', 'Slippage %']
        
        for _, row in table_df.iterrows():
            recommendation = row.get('Recommendation', '')
            row_class = f"recommendation-{recommendation}" if recommendation in ['strong_buy', 'buy', 'consider', 'avoid'] else ""
            html_content += f'<tr class="{row_class}">'
            
            for col_name, value in zip(table_df.columns, row):
                if col_name in numeric_columns and value != "N/A":
                    # Extract numeric value for sorting
                    if isinstance(value, str):
                        # Remove formatting to get raw number
                        numeric_value = value.replace(',', '').replace(' gp', '').replace('%', '')
                        try:
                            numeric_value = float(numeric_value)
                            html_content += f'<td data-order="{numeric_value}">{value}</td>'
                        except ValueError:
                            html_content += f'<td>{value}</td>'
                    else:
                        html_content += f'<td data-order="{value}">{value}</td>'
                else:
                    html_content += f'<td>{value}</td>'
            html_content += '</tr>'
        
        html_content += f"""
                    </tbody>
                </table>
            </div>
            
            <script>
                $(document).ready(function() {{
                    $('#priceTable').DataTable({{
                        "pageLength": 50,
                        "lengthMenu": [[25, 50, 100, -1], [25, 50, 100, "All"]],
                        "order": [[ 4, "desc" ]], // Sort by Decline % descending by default
                        "columnDefs": [
                            {{
                                "targets": [1, 2, 3, 4, 5, 8, 9, 10, 11, 15, 16, 17], // All numeric columns
                                "type": "num",
                                "render": function(data, type, row) {{
                                    if (type === 'display' || type === 'type') {{
                                        return data;
                                    }}
                                    // For sorting, use the data-order attribute if available
                                    return data;
                                }}
                            }},
                            {{
                                "targets": [4], // Decline % column
                                "className": "text-center font-weight-bold"
                            }}
                        ],
                        "scrollX": true,
                        "responsive": true,
                        "language": {{
                            "search": "Search items:",
                            "lengthMenu": "Show _MENU_ items per page",
                            "info": "Showing _START_ to _END_ of _TOTAL_ items",
                            "paginate": {{
                                "first": "First",
                                "last": "Last",
                                "next": "Next",
                                "previous": "Previous"
                            }}
                        }}
                    }});
                    
                    // Add click handlers for easy sorting
                    $('.dataTables_wrapper').prepend(`
                        <div style="margin-bottom: 15px; padding: 10px; background-color: #f8f9fa; border-radius: 5px;">
                            <strong>Quick Sort:</strong>
                            <button onclick="sortTable(4, 'desc')" class="btn btn-sm btn-primary" style="margin: 2px;">Highest Decline %</button>
                            <button onclick="sortTable(5, 'desc')" class="btn btn-sm btn-success" style="margin: 2px;">Best Net Profit %</button>
                            <button onclick="sortTable(11, 'desc')" class="btn btn-sm btn-info" style="margin: 2px;">Highest Opportunity</button>
                            <button onclick="sortTable(8, 'desc')" class="btn btn-sm btn-warning" style="margin: 2px;">Best Liquidity</button>
                            <button onclick="sortTable(15, 'desc')" class="btn btn-sm btn-secondary" style="margin: 2px;">Highest Volume</button>
                        </div>
                    `);
                }});
                
                function sortTable(columnIndex, direction) {{
                    var table = $('#priceTable').DataTable();
                    table.order([columnIndex, direction]).draw();
                }}
            </script>
        </body>
        </html>
        """
        
        # Save the HTML table
        timestamp_file = datetime.now().strftime("%Y%m%d_%H%M%S")
        filepath = f"{EXPORTS_PATH}/sortable_table_{timestamp_file}.html"
        
        os.makedirs(EXPORTS_PATH, exist_ok=True)
        
        with open(filepath, 'w', encoding='utf-8') as f:
            f.write(html_content)
        
        logger.info(f"Sortable HTML table saved to {filepath}")
        return filepath
    
    def create_comprehensive_dashboard(self, results_df: pd.DataFrame, summary_stats: Dict) -> str:
        """
        Create comprehensive HTML dashboard with multiple interactive components
        
        Args:
            results_df: DataFrame with analysis results
            summary_stats: Dictionary with summary statistics
        
        Returns:
            Path to saved HTML dashboard
        """
        if results_df.empty:
            logger.warning("No data available for dashboard creation")
            return ""
        
        # Create individual components and save them as separate HTML files
        data_table = self.create_interactive_data_table(results_df)
        decline_chart = self.create_decline_bar_chart(results_df, 30)
        distribution_chart = self.create_decline_distribution(results_df)
        scatter_fig = self.create_liquidity_opportunity_scatter(results_df)
        quality_fig = self.create_quality_analysis_chart(results_df)
        
        # Save individual charts to embed in dashboard
        timestamp_file = datetime.now().strftime("%Y%m%d_%H%M%S")
        os.makedirs(EXPORTS_PATH, exist_ok=True)
        
        data_table_file = f"{EXPORTS_PATH}/temp_data_table_{timestamp_file}.html"
        decline_chart_file = f"{EXPORTS_PATH}/temp_decline_chart_{timestamp_file}.html"
        distribution_chart_file = f"{EXPORTS_PATH}/temp_distribution_chart_{timestamp_file}.html"
        scatter_chart_file = f"{EXPORTS_PATH}/temp_scatter_chart_{timestamp_file}.html"
        quality_chart_file = f"{EXPORTS_PATH}/temp_quality_chart_{timestamp_file}.html"
        
        data_table.write_html(data_table_file, include_plotlyjs='cdn')
        decline_chart.write_html(decline_chart_file, include_plotlyjs='cdn')
        distribution_chart.write_html(distribution_chart_file, include_plotlyjs='cdn')
        scatter_fig.write_html(scatter_chart_file, include_plotlyjs='cdn')
        quality_fig.write_html(quality_chart_file, include_plotlyjs='cdn')
        
        # Generate HTML content
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        
        html_content = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <title>OSRS Price Analysis Dashboard</title>
            <script src="https://cdn.plot.ly/plotly-latest.min.js"></script>
            <style>
                body {{
                    font-family: Arial, sans-serif;
                    margin: 20px;
                    background-color: #f5f5f5;
                }}
                .header {{
                    text-align: center;
                    background-color: #2c3e50;
                    color: white;
                    padding: 20px;
                    border-radius: 10px;
                    margin-bottom: 20px;
                }}
                .summary-stats {{
                    display: grid;
                    grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
                    gap: 15px;
                    margin-bottom: 30px;
                }}
                .stat-card {{
                    background: white;
                    padding: 15px;
                    border-radius: 8px;
                    box-shadow: 0 2px 4px rgba(0,0,0,0.1);
                    text-align: center;
                }}
                .stat-value {{
                    font-size: 24px;
                    font-weight: bold;
                    color: #3498db;
                }}
                .stat-label {{
                    color: #7f8c8d;
                    font-size: 14px;
                }}
                .chart-container {{
                    background: white;
                    margin-bottom: 30px;
                    padding: 20px;
                    border-radius: 10px;
                    box-shadow: 0 2px 8px rgba(0,0,0,0.1);
                }}
                .filters {{
                    background: white;
                    padding: 20px;
                    border-radius: 10px;
                    margin-bottom: 20px;
                    box-shadow: 0 2px 4px rgba(0,0,0,0.1);
                }}
                .filter-group {{
                    display: inline-block;
                    margin-right: 20px;
                    margin-bottom: 10px;
                }}
                .filter-group label {{
                    display: block;
                    font-weight: bold;
                    margin-bottom: 5px;
                }}
                .filter-group select, .filter-group input {{
                    padding: 5px;
                    border: 1px solid #ddd;
                    border-radius: 4px;
                }}
            </style>
        </head>
        <body>
            <div class="header">
                <h1>🏰 OSRS Price Analysis Dashboard</h1>
                <p>Comprehensive analysis with liquidity, stability, and economic viability metrics</p>
                <p><small>Generated: {timestamp}</small></p>
            </div>
            
            <div class="summary-stats">
                <div class="stat-card">
                    <div class="stat-value">{summary_stats.get('total_items', 0):,}</div>
                    <div class="stat-label">Total Items</div>
                </div>
                <div class="stat-card">
                    <div class="stat-value">{summary_stats.get('enhanced_viable_items', 0):,}</div>
                    <div class="stat-label">Enhanced Viable</div>
                </div>
                <div class="stat-card">
                    <div class="stat-value">{summary_stats.get('enhanced_strong_buy', 0):,}</div>
                    <div class="stat-label">Strong Buy</div>
                </div>
                <div class="stat-card">
                    <div class="stat-value">{summary_stats.get('stable_market_items', 0):,}</div>
                    <div class="stat-label">Stable Markets</div>
                </div>
                <div class="stat-card">
                    <div class="stat-value">{summary_stats.get('genuine_decline_items', 0):,}</div>
                    <div class="stat-label">Genuine Declines</div>
                </div>
                <div class="stat-card">
                    <div class="stat-value">{summary_stats.get('average_liquidity_score', 0):.0f}</div>
                    <div class="stat-label">Avg Liquidity Score</div>
                </div>
            </div>
            
            <div class="chart-container">
                <h2>📊 Interactive Data Table</h2>
                <p>Click column headers to sort. Scroll horizontally to see all columns.</p>
                <iframe src="{os.path.basename(data_table_file)}" width="100%" height="800" frameborder="0"></iframe>
            </div>
            
            <div class="chart-container">
                <h2>📉 Top Declining Items</h2>
                <iframe src="{os.path.basename(decline_chart_file)}" width="100%" height="600" frameborder="0"></iframe>
            </div>
            
            <div class="chart-container">
                <h2>💰 Liquidity vs Opportunity Analysis</h2>
                <p>Green = Strong Buy, Yellow = Consider, Red = Avoid. Larger dots = higher quality scores.</p>
                <iframe src="{os.path.basename(scatter_chart_file)}" width="100%" height="600" frameborder="0"></iframe>
            </div>
            
            <div class="chart-container">
                <h2>🔍 Quality Analysis</h2>
                <iframe src="{os.path.basename(quality_chart_file)}" width="100%" height="800" frameborder="0"></iframe>
            </div>
            
            <div class="chart-container">
                <h2>📈 Price Decline Distribution</h2>
                <iframe src="{os.path.basename(distribution_chart_file)}" width="100%" height="600" frameborder="0"></iframe>
            </div>
        </body>
        </html>
        """
        
        # Save dashboard
        timestamp_file = datetime.now().strftime("%Y%m%d_%H%M%S")
        filepath = f"{EXPORTS_PATH}/osrs_dashboard_{timestamp_file}.html"
        
        # Ensure directory exists
        os.makedirs(EXPORTS_PATH, exist_ok=True)
        
        with open(filepath, 'w', encoding='utf-8') as f:
            f.write(html_content)
        
        logger.info(f"Comprehensive dashboard saved to {filepath}")
        return filepath
    
    def create_liquidity_opportunity_scatter(self, results_df: pd.DataFrame) -> go.Figure:
        """
        Create scatter plot showing liquidity vs opportunity scores
        
        Args:
            results_df: DataFrame with analysis results
        
        Returns:
            Plotly figure object
        """
        if results_df.empty:
            return go.Figure()
        
        # Color by recommendation
        color_map = {
            'strong_buy': '#00ff00',
            'buy': '#90EE90',
            'consider': '#FFD700', 
            'avoid': '#FF6B6B'
        }
        
        colors = [color_map.get(rec, '#CCCCCC') for rec in results_df['enhanced_recommendation']]
        
        fig = go.Figure()
        
        # Handle NaN values in quality_score for marker sizing
        quality_scores = results_df['quality_score'].fillna(50)  # Default to 50 if NaN
        marker_sizes = quality_scores / 5
        marker_sizes = marker_sizes.clip(lower=5, upper=20)  # Ensure reasonable size range
        
        fig.add_trace(go.Scatter(
            x=results_df['liquidity_score'],
            y=results_df['adjusted_opportunity_score'],
            mode='markers',
            marker=dict(
                color=colors,
                size=marker_sizes,  # Use cleaned sizes
                opacity=0.7,
                line=dict(width=1, color='black')
            ),
            text=results_df['item_name'],
            hovertemplate=(
                "<b>%{text}</b><br>" +
                "Liquidity Score: %{x:.0f}<br>" +
                "Opportunity Score: %{y:.0f}<br>" +
                "Quality Score: %{customdata[0]:.0f}<br>" +
                "Recommendation: %{customdata[1]}<br>" +
                "Net Profit: %{customdata[2]:.1f}%<br>" +
                "<extra></extra>"
            ),
            customdata=results_df[['quality_score', 'enhanced_recommendation', 'net_profit_percentage']].values
        ))
        
        # Add quadrant lines
        fig.add_hline(y=50, line_dash="dash", line_color="gray", opacity=0.5)
        fig.add_vline(x=50, line_dash="dash", line_color="gray", opacity=0.5)
        
        # Add quadrant labels
        fig.add_annotation(x=75, y=75, text="High Liquidity<br>High Opportunity", showarrow=False, 
                          bgcolor="rgba(0,255,0,0.1)", bordercolor="green")
        fig.add_annotation(x=25, y=75, text="Low Liquidity<br>High Opportunity", showarrow=False,
                          bgcolor="rgba(255,255,0,0.1)", bordercolor="orange")
        fig.add_annotation(x=75, y=25, text="High Liquidity<br>Low Opportunity", showarrow=False,
                          bgcolor="rgba(255,255,0,0.1)", bordercolor="orange")
        fig.add_annotation(x=25, y=25, text="Low Liquidity<br>Low Opportunity", showarrow=False,
                          bgcolor="rgba(255,0,0,0.1)", bordercolor="red")
        
        fig.update_layout(
            title="Liquidity vs Opportunity Analysis",
            xaxis_title="Liquidity Score",
            yaxis_title="Adjusted Opportunity Score",
            width=800,
            height=600
        )
        
        return fig
    
    def create_quality_analysis_chart(self, results_df: pd.DataFrame) -> go.Figure:
        """
        Create chart analyzing quality metrics
        
        Args:
            results_df: DataFrame with analysis results
        
        Returns:
            Plotly figure object
        """
        if results_df.empty:
            return go.Figure()
        
        # Create subplots for quality analysis
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=(
                "Stability Score Distribution",
                "Trend Types",
                "Execution Feasibility", 
                "Quality vs Profit"
            ),
            specs=[
                [{"type": "histogram"}, {"type": "pie"}],
                [{"type": "pie"}, {"type": "scatter"}]
            ]
        )
        
        # Stability score distribution
        fig.add_trace(
            go.Histogram(x=results_df['stability_score'], nbinsx=20, name="Stability"),
            row=1, col=1
        )
        
        # Trend types pie chart
        trend_counts = results_df['trend_type'].value_counts()
        fig.add_trace(
            go.Pie(labels=trend_counts.index, values=trend_counts.values, name="Trends"),
            row=1, col=2
        )
        
        # Execution feasibility
        exec_counts = results_df['execution_feasibility'].value_counts()
        fig.add_trace(
            go.Pie(labels=exec_counts.index, values=exec_counts.values, name="Execution"),
            row=2, col=1
        )
        
        # Quality vs Profit scatter
        fig.add_trace(
            go.Scatter(
                x=results_df['quality_score'],
                y=results_df['net_profit_percentage'],
                mode='markers',
                name="Quality vs Profit",
                text=results_df['item_name']
            ),
            row=2, col=2
        )
        
        fig.update_layout(
            title="Quality Analysis Dashboard",
            width=1200,
            height=800,
            showlegend=False
        )
        
        return fig
