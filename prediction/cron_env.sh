#!/bin/bash
for variable_value in $(cat /proc/1/environ | sed 's/\x00/\n/g'); do
  export $variable_value
done