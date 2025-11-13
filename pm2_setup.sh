#!/bin/bash

# PM2 Setup and Management Script for Video Clip Manager

# Colors for output
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m' # No Color

echo -e "${BLUE}==================================${NC}"
echo -e "${BLUE}Video Clip Manager - PM2 Setup${NC}"
echo -e "${BLUE}==================================${NC}"

# Create logs directory
echo -e "\n${GREEN}Creating logs directory...${NC}"
mkdir -p /var/www/spera_AI/6_pose/pose_detection/logs

# Check if PM2 is installed
if ! command -v pm2 &> /dev/null
then
    echo -e "${YELLOW}PM2 is not installed. Installing PM2...${NC}"
    npm install -g pm2
    echo -e "${GREEN}PM2 installed successfully!${NC}"
else
    echo -e "${GREEN}PM2 is already installed.${NC}"
fi

# Navigate to project directory
cd /var/www/spera_AI/6_pose/pose_detection

echo -e "\n${BLUE}Available PM2 commands:${NC}"
echo -e "${GREEN}1. Start application${NC}"
echo -e "${GREEN}2. Stop application${NC}"
echo -e "${GREEN}3. Restart application${NC}"
echo -e "${GREEN}4. View logs${NC}"
echo -e "${GREEN}5. View status${NC}"
echo -e "${GREEN}6. Setup startup script${NC}"
echo -e "${GREEN}7. Delete from PM2${NC}"

echo -e "\n${YELLOW}Quick commands:${NC}"
echo -e "Start:    ${BLUE}pm2 start pm2.config.json${NC}"
echo -e "Stop:     ${BLUE}pm2 stop video-clip-manager${NC}"
echo -e "Restart:  ${BLUE}pm2 restart video-clip-manager${NC}"
echo -e "Logs:     ${BLUE}pm2 logs video-clip-manager${NC}"
echo -e "Status:   ${BLUE}pm2 status${NC}"
echo -e "Monitor:  ${BLUE}pm2 monit${NC}"
