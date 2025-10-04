#!/bin/bash
set -euo pipefail

CURRENT_DIR=$(pwd)
BASE_DIR=$(dirname $(cd "$(dirname "$0")" && pwd))
BUILD_DIR="$BASE_DIR/build"

RED='\033[1;31m'
GREEN='\033[1;32m'
YELLOW='\033[1;33m'
BLUE='\033[1;34m'
MAGENTA='\033[1;35m'
CYAN='\033[1;36m'
GRAY='\033[1;90m'
NC='\033[0m' 
BOLD='\033[1m'
DIM='\033[2m'
SPACE='                                                             '

show_header() {
  echo -e "${DIM}                HPC Cmake Project Builder${NC}"
  echo -e "${DIM}──────────────────────────────────────────────────────────${NC}"
  echo
}

select_cmake_generator() {
    local generator=""
    local options=("Default" "Ninja" "Custom")
    local current_selection=0
    
    echo -e "${DIM}${BLUE}◇ Select CMake Generator${NC}" >&2
    echo -e "│${DIM} Use arrow keys to navigate, Enter to confirm${NC}" >&2
    
    local menu_lines=0
    
    display_menu() {
        for i in "${!options[@]}"; do
            if [ $i -eq "$current_selection" ]; then
                echo -e "│ ${BOLD}${GREEN}❯ ${options[i]}${NC}${SPACE}" >&2
            else
                echo -e "│ ${DIM}${GRAY}${options[i]}${NC}${SPACE}" >&2
            fi
        done
        menu_lines="${#options[@]}"
    }
    
    display_menu
    
    echo -ne "\033[s" >&2
    
    while true; do
        read -rsn1 key
        
        if [ "$key" = $'\x1b' ]; then
            read -rsn1 -t 1 key2
            if [ "$key2" = '[' ]; then
                read -rsn1 key3
                case "$key3" in
                    'A')
                        if [ "$current_selection" -gt 0 ]; then
                            current_selection="$((current_selection - 1))"
                            echo -ne "\033[u" >&2
                            echo -ne "\033[${menu_lines}A\033[J" >&2
                            display_menu
                        fi
                        ;;
                    'B') 
                        if [ "$current_selection" -lt "$((${#options[@]} - 1))" ]; then
                            current_selection="$((current_selection + 1))"
                            echo -ne "\033[u" >&2
                            echo -ne "\033[${menu_lines}A\033[J" >&2
                            display_menu
                        fi
                        ;;
                esac
            fi
        elif [ "$key" = "" ]; then
            break
        fi
    done
    
    echo -ne "\033[u\033[J" >&2 
    
    case "${options[current_selection]}" in
        "Default") 
            generator=""
            ;;
        "Custom")
            echo -n "Please enter a custom build type: " >&2
            read custom_bt
            ;;
        *)
            generator="${options[current_selection]}"
            ;;
    esac
    
    echo -e "\033[7A\033[2K${DIM}${BLUE}◆ Generator Selected!${NC}${SPACE}" >&2
    display_menu
    echo -e "${GREEN}✔ Generator: ${NC}${BOLD}$generator${NC}" >&2
    
    echo "$generator"
}

bootstrap() {
  show_header
  generator=$(select_cmake_generator)

  # Build script for a CMake project
  mkdir -p $BUILD_DIR
  cd $BUILD_DIR
  if [ -z "$generator" ]; then
    cmake "$BASE_DIR"
  else
    cmake -G "$generator" "$BASE_DIR"
  fi

  cd $BASE_DIR
}

bootstrap
