#!/bin/bash

# Check machine's IP type (DHCP, etc)
iface=$(ip route show default | awk '/default/ {print $5}')
ip_type=$(grep -w "$iface" /etc/network/interfaces | grep -oP '(?<=inet\s).*(?=\sstatic|dhcp)' || echo "unknown")

# Check machine's MAC address
mac_address=$(cat /sys/class/net/$iface/address)

# Check machine's hardware
# CPU
cpu_info=$(lscpu | grep "Model name" | awk -F: '{print $2}' | sed 's/^[[:blank:]]*//')

# RAM
ram_info=$(free -h | awk '/^Mem:/ {print $2}')

# Disk Capacity
disk_capacity=$(df -h --total | awk 'END{print $2}')

# Operational System
os_info=$(cat /etc/os-release | grep "^PRETTY_NAME" | awk -F= '{print $2}' | sed 's/^"//' | sed 's/"$//')

# Save all this information as a list
info_list=("IP Type: $ip_type" "MAC Address: $mac_address" "CPU: $cpu_info" "RAM: $ram_info" "Disk Capacity: $disk_capacity" "OS: $os_info")

# Print the list
for item in "${info_list[@]}"; do
    echo "$item"
done

# Save the list to a text file
printf "%s\n" "${info_list[@]}" > machine_info.txt
