cat mem.log | awk -F ',' '{print $2}' | uniq -u | sort
