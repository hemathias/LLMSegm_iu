#!/bin/bash

# Check if input file is provided
if [ $# -ne 1 ]; then
  echo "Usage: $0 <input.csv>"
  exit 1
fi

input_file="$1"

# Calculate the total number of lines in the input file
total_lines=$(wc -l < "$input_file")

# Calculate the number of lines for each output file
train_lines=$(echo "0.8 * $total_lines" | bc | awk '{print int($1+0.5)}')
dev_lines=$(echo "0.1 * $total_lines" | bc | awk '{print int($1+0.5)}')
test_lines=$(echo "0.1 * $total_lines" | bc | awk '{print int($1+0.5)}')

echo "Train lines before leftover adding: $train_lines"

# Calculate total of split lines and leftover lines
split_total=$((train_lines + dev_lines + test_lines))
leftover_lines=$((total_lines - split_total))

# Add leftover lines to the train.csv file
train_lines=$((train_lines + leftover_lines))

echo "Train lines after leftover adding: $train_lines"

# Shuffle the input file to ensure random distribution
shuf "$input_file" > shuffled.csv

# Split the shuffled file into train, dev, and test files
head -n "$train_lines" shuffled.csv > train.iu.csv
tail -n +"$((train_lines + 1))" shuffled.csv | head -n "$dev_lines" > dev.iu.csv
tail -n "$test_lines" shuffled.csv > test.iu.csv

# Clean up the temporary shuffled file
rm shuffled.csv

echo "Splitting complete:"
echo "train.csv: $train_lines lines"
echo "dev.csv: $dev_lines lines"
echo "test.csv: $test_lines lines"
