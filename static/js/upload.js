// MetaRecom CSV upload and target-column detection.

(function () {

  const form = document.getElementById("run-form");
  const btn = document.getElementById("submit-btn");
  const loading = document.getElementById("loading-msg");

  const fileIn = document.getElementById("file");
  const samples = document.querySelectorAll('input[name="sample"]');

  const targetSelect = document.getElementById("target_column");
  const targetHelp = document.getElementById("target-help");

  const customToggle = document.getElementById("custom-target-toggle");
  const customTarget = document.getElementById("custom_target_column");

  if (!form) return;


  // Reset the target selector
  function resetTargetSelector() {

    if (targetSelect) {
      targetSelect.innerHTML =
        '<option value="">Upload a CSV to detect columns</option>';

      targetSelect.disabled = true;
    }

    if (targetHelp) {
      targetHelp.textContent =
        "Upload a CSV and MetaRecom will inspect the dataset and recommend a target column.";
    }

    if (customToggle) {
      customToggle.checked = false;
    }

    if (customTarget) {
      customTarget.value = "";
      customTarget.disabled = true;
    }
  }


  // Inspect the uploaded CSV
  async function inspectCSV(file) {

    if (!file) {
      resetTargetSelector();
      return;
    }


    if (!file.name.toLowerCase().endsWith(".csv")) {

      if (targetSelect) {
        targetSelect.innerHTML =
          '<option value="">Please upload a CSV file</option>';
      }

      if (targetHelp) {
        targetHelp.textContent =
          "Only CSV files can be inspected.";
      }

      return;
    }


    // Loading state
    if (targetSelect) {

      targetSelect.disabled = true;

      targetSelect.innerHTML =
        '<option value="">Detecting columns...</option>';
    }

    if (targetHelp) {
      targetHelp.textContent =
        "Inspecting your CSV...";
    }


    try {

      const formData = new FormData();

      formData.append("file", file);


      const response = await fetch("/api/inspect-csv", {
        method: "POST",
        body: formData
      });


      // Read as text first so we can safely handle
      // both JSON and accidental HTML error pages.
      const responseText = await response.text();

      let data;

      try {
        data = JSON.parse(responseText);
      } catch (jsonError) {

        console.error(
          "Server returned non-JSON response:",
          responseText
        );

        throw new Error(
          "The CSV inspection endpoint did not return JSON. " +
          "Make sure /api/inspect-csv exists in app.py and restart Flask."
        );
      }


      if (!response.ok) {

        throw new Error(
          data.error || "Could not inspect the CSV."
        );
      }


      const columns = Array.isArray(data.columns)
        ? data.columns
        : [];

      const suggestedTarget = data.suggested_target;


      if (!targetSelect) return;


      targetSelect.innerHTML = "";


      if (columns.length === 0) {

        targetSelect.innerHTML =
          '<option value="">No columns detected</option>';

        targetSelect.disabled = true;

        if (targetHelp) {
          targetHelp.textContent =
            "No usable columns were found in this CSV.";
        }

        return;
      }


      // Populate dropdown with detected columns
      columns.forEach((column) => {

        const option = document.createElement("option");

        option.value = column;
        option.textContent = column;

        if (column === suggestedTarget) {
          option.selected = true;
        }

        targetSelect.appendChild(option);

      });


      targetSelect.disabled = false;


      // Show recommendation
      if (targetHelp) {

        if (suggestedTarget) {

          targetHelp.textContent =
            `Recommended target column: ${suggestedTarget}`;

        } else {

          targetHelp.textContent =
            "Select the column you want MetaRecom to predict.";
        }
      }


    } catch (error) {

      console.error("CSV inspection failed:", error);


      if (targetSelect) {

        targetSelect.innerHTML =
          '<option value="">Could not detect columns</option>';

        targetSelect.disabled = true;
      }


      if (targetHelp) {

        targetHelp.textContent =
          `Could not inspect this CSV: ${error.message}`;
      }

    }

  }


  // When user uploads a file
  if (fileIn) {

    fileIn.addEventListener("change", async function () {

      if (fileIn.files && fileIn.files.length > 0) {

        // Unselect sample datasets
        samples.forEach((sample) => {
          sample.checked = false;
        });

        await inspectCSV(fileIn.files[0]);

      } else {

        resetTargetSelector();

      }

    });

  }


  // When user selects a sample dataset
  samples.forEach((sample) => {

    sample.addEventListener("change", function () {

      if (!sample.checked) return;

      // Clear uploaded file
      if (fileIn) {
        fileIn.value = "";
      }

      resetTargetSelector();

    });

  });


  // Manual target override
  if (customToggle && customTarget) {

    customToggle.addEventListener("change", function () {

      customTarget.disabled = !customToggle.checked;

      if (customToggle.checked) {
        customTarget.focus();
      } else {
        customTarget.value = "";
      }

    });

  }


  // Submit form
  form.addEventListener("submit", function () {

    // If the user manually entered a target,
    // add/select it in the actual submitted field.
    if (
      customToggle &&
      customToggle.checked &&
      customTarget &&
      customTarget.value.trim() &&
      targetSelect
    ) {

      const manualTarget = customTarget.value.trim();

      let optionExists = false;

      Array.from(targetSelect.options).forEach((option) => {

        if (option.value === manualTarget) {
          optionExists = true;
        }

      });


      if (!optionExists) {

        const option = document.createElement("option");

        option.value = manualTarget;
        option.textContent = manualTarget;

        targetSelect.appendChild(option);

      }


      targetSelect.value = manualTarget;
      targetSelect.disabled = false;

    }


    // Make sure detected target is submitted.
    if (targetSelect && targetSelect.disabled) {
      targetSelect.disabled = false;
    }


    if (btn) {
      btn.disabled = true;
      btn.textContent = "Running...";
    }


    if (loading) {
      loading.hidden = false;
    }

  });


})();