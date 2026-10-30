// Plotly charts for the results page.
// Reads window.__RESULT__ injected by results.html.

(function () {
  const resultElement = document.getElementById('result-data');

  if (!resultElement) return;

  let result;

  try {
    result = JSON.parse(resultElement.textContent);
  } catch (error) {
    console.error('Could not parse result data:', error);
    return;
  }


  // ============================================================
  // THEME HELPERS
  // ============================================================

  const cssVar = (name) =>
    getComputedStyle(document.documentElement)
      .getPropertyValue(name)
      .trim();


  const layoutBase = () => ({

    paper_bgcolor: 'rgba(0,0,0,0)',

    plot_bgcolor: 'rgba(0,0,0,0)',

    font: {
      color: cssVar('--text') || '#CCD6F6',
      family: 'inherit'
    },

    margin: {
      l: 50,
      r: 20,
      t: 20,
      b: 50
    }

  });


  const plotConfig = {
    displayModeBar: false,
    responsive: true
  };


  // ============================================================
  // MODEL COMPARISON
  // ============================================================

  const modelsEl = document.getElementById('chart-models');


  if (
    modelsEl &&
    Array.isArray(result.all_results) &&
    result.all_results.length
  ) {

    const rows = result.all_results;


    const scoreKey =
      ('accuracy' in rows[0])
        ? 'accuracy'
        : ('r2' in rows[0])
          ? 'r2'
          : Object.keys(rows[0])
              .find((k) => typeof rows[0][k] === 'number');


    const nameKey =
      ('model' in rows[0])
        ? 'model'
        : ('model_name' in rows[0])
          ? 'model_name'
          : Object.keys(rows[0])[0];


    const sorted = [...rows].sort(
      (a, b) =>
        (b[scoreKey] ?? 0) -
        (a[scoreKey] ?? 0)
    );


    const names = sorted.map(
      (r) => r[nameKey]
    );


    const scores = sorted.map(
      (r) => r[scoreKey] ?? 0
    );


    const best = scores[0];


    const colors = scores.map(
      (score) =>

        score === best
          ? cssVar('--primary') || '#6C63FF'
          : cssVar('--accent') || '#FF6584'
    );


    Plotly.newPlot(

      modelsEl,

      [{

        type: 'bar',

        x: scores,

        y: names,

        orientation: 'h',

        marker: {
          color: colors
        },

        hovertemplate:
          '%{y}: %{x:.4f}<extra></extra>'

      }],

      {

        ...layoutBase(),

        xaxis: {

          title: scoreKey,

          gridcolor:
            cssVar('--border')

        },

        yaxis: {

          autorange: 'reversed'

        }

      },

      plotConfig

    );

  }


  // ============================================================
  // META-FEATURE RADAR
  // ============================================================

  const radarEl =
    document.getElementById('chart-radar');


  if (
    radarEl &&
    result.meta_features
  ) {

    const interesting = [

      'log_n_samples',

      'log_n_features',

      'samples_per_feature',

      'ratio_numeric',

      'ratio_categorical',

      'missing_ratio_overall',

      'mean_skewness',

      'mean_kurtosis',

      'class_imbalance',

      'n_classes'

    ];


    const labels = [];

    const values = [];


    interesting.forEach((key) => {

      if (
        key in result.meta_features &&
        result.meta_features[key] != null
      ) {

        labels.push(key);


        let value =
          Number(result.meta_features[key]);


        if (!isFinite(value)) {

          value = 0;

        }


        // Squash values to roughly 0–1
        // so the radar chart stays readable.

        value =
          Math.tanh(
            Math.abs(value) / 5
          );


        values.push(value);

      }

    });


    if (labels.length >= 3) {

      labels.push(labels[0]);

      values.push(values[0]);


      Plotly.newPlot(

        radarEl,

        [{

          type: 'scatterpolar',

          r: values,

          theta: labels,

          fill: 'toself',

          line: {

            color:
              cssVar('--primary') ||
              '#6C63FF'

          },

          fillcolor:
            'rgba(108,99,255,0.25)',

          hovertemplate:
            '%{theta}: %{r:.2f}<extra></extra>'

        }],

        {

          ...layoutBase(),

          polar: {

            bgcolor:
              'rgba(0,0,0,0)',

            radialaxis: {

              visible: true,

              range: [0, 1],

              gridcolor:
                cssVar('--border')

            },

            angularaxis: {

              gridcolor:
                cssVar('--border')

            }

          },

          showlegend: false

        },

        plotConfig

      );

    } else {

      radarEl.innerHTML =

        '<p class="muted" ' +
        'style="text-align:center;padding:1rem;">' +
        'Not enough meta-features to render radar.' +
        '</p>';

    }

  }


  // ============================================================
  // PREPROCESSING: BEFORE VS AFTER
  // ============================================================

  const preprocessingBeforeAfterEl =
    document.getElementById(
      'chart-preprocessing-before-after'
    );


  const preprocessing =
    result.preprocessing;


  if (
    preprocessingBeforeAfterEl &&
    preprocessing &&
    preprocessing.applied &&
    preprocessing.cleaning_summary
  ) {

    const summary =
      preprocessing.cleaning_summary;


    const original =
      summary.original_dataset || {};


    const processed =
      summary.processed_dataset || {};


    const categories = [

      'Rows',

      'Columns',

      'Missing Values'

    ];


    const before = [

      Number(original.rows || 0),

      Number(original.columns || 0),

      Number(original.missing_values || 0)

    ];


    const after = [

      Number(processed.rows || 0),

      Number(processed.columns || 0),

      Number(processed.missing_values || 0)

    ];


    Plotly.newPlot(

      preprocessingBeforeAfterEl,

      [

        {

          type: 'bar',

          name: 'Before',

          x: categories,

          y: before,

          hovertemplate:
            '%{x}: %{y}<extra>Before</extra>'

        },

        {

          type: 'bar',

          name: 'After',

          x: categories,

          y: after,

          hovertemplate:
            '%{x}: %{y}<extra>After</extra>'

        }

      ],

      {

        ...layoutBase(),

        barmode: 'group',

        xaxis: {

          title: 'Dataset property'

        },

        yaxis: {

          title: 'Count',

          gridcolor:
            cssVar('--border')

        },

        legend: {

          orientation: 'h',

          y: 1.15

        }

      },

      plotConfig

    );

  }


  // ============================================================
  // PREPROCESSING ACTIONS
  // ============================================================

  const preprocessingActionsEl =
    document.getElementById(
      'chart-preprocessing-actions'
    );


  if (
    preprocessingActionsEl &&
    preprocessing &&
    preprocessing.applied &&
    preprocessing.cleaning_summary &&
    preprocessing.cleaning_summary.cleaning_actions
  ) {

    const actions =
      preprocessing.cleaning_summary
        .cleaning_actions;


    const labels = [];

    const values = [];


    const addAction =
      (label, value) => {

        const numericValue =
          Array.isArray(value)
            ? value.length
            : Number(value || 0);


        if (numericValue > 0) {

          labels.push(label);

          values.push(numericValue);

        }

      };


    addAction(
      'Missing Values Normalized',
      actions.normalized_missing_values
    );


    addAction(
      'Invalid Numeric Values',
      actions.invalid_numeric_values_converted
    );


    addAction(
      'Infinite Values',
      actions.infinite_values_replaced
    );


    addAction(
      'Rows Removed (Missing Target)',
      actions.rows_removed_missing_target
    );


    addAction(
      'Duplicate Rows Removed',
      actions.duplicate_rows_removed
    );


    addAction(
      'Empty Columns Removed',
      actions.empty_columns_removed
    );


    addAction(
      'Constant Columns Removed',
      actions.constant_columns_removed
    );


    if (labels.length > 0) {

      Plotly.newPlot(

        preprocessingActionsEl,

        [{

          type: 'bar',

          x: values,

          y: labels,

          orientation: 'h',

          marker: {

            color:
              cssVar('--primary') ||
              '#6C63FF'

          },

          hovertemplate:
            '%{y}: %{x}<extra></extra>'

        }],

        {

          ...layoutBase(),

          xaxis: {

            title: 'Number of corrections',

            gridcolor:
              cssVar('--border')

          },

          yaxis: {

            autorange: 'reversed'

          }

        },

        plotConfig

      );

    } else {

      preprocessingActionsEl.innerHTML =

        '<p class="muted" ' +
        'style="text-align:center;padding:4rem 1rem;">' +
        'No preprocessing corrections were required.' +
        '</p>';

    }

  }


  // ============================================================
  // THEME CHANGE SUPPORT
  // ============================================================

  const chartElements = [

    modelsEl,

    radarEl,

    preprocessingBeforeAfterEl,

    preprocessingActionsEl

  ];


  const obs =
    new MutationObserver(() => {

      chartElements.forEach(
        (chart) => {

          if (chart?.data) {

            Plotly.relayout(
              chart,
              layoutBase()
            );

          }

        }
      );

    });


  obs.observe(

    document.documentElement,

    {

      attributes: true,

      attributeFilter: ['data-theme']

    }

  );

})();