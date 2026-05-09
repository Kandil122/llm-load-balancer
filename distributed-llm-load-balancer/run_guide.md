# Distributed LLM Load Balancer — Docker Run Guide

### 🚀 HOW TO RUN THE PROJECT

#### 1. Navigate to Project Directory
```bash
cd "/storage/UNIcourses/Distributed Computing/project/distributed-llm-load-balancer/distributed-llm-load-balancer/"
```

#### 2. Activate Virtual Environment
```bash
source ../.venv/bin/activate
```

# Run the Main Script (Simulation or Distributed)
The `main.py` script can run a local simulation OR act as a client for the Docker containers.

```bash
# 1. Local Simulation Mode (Default)
python main.py --strategy round_robin --users 10 --workers 4

# 2. Distributed Mode (Hits the Docker containers)
# Make sure "docker compose up" is running first!
python main.py --distributed --users 10 --workers 4

# Run with all options
python main.py --distributed --strategy [STRATEGY] --users [COUNT] --workers [COUNT] --no-fault --save-results
```

**Available Options:**
| Option | Description | Default |
| :--- | :--- | :--- |
| `--distributed` | Run against Docker containers (Master at localhost:8000) | `False` |
| `--strategy` | Load balancing strategy: `round_robin`, `least_connections`, or `load_aware` | `round_robin` |

| `--users` | Number of concurrent users to simulate | `20` (from .env) |
| `--workers` | Number of worker instances to create | `4` (from .env) |
| `--no-fault` | Disable the worker failure simulation | `False` |
| `--save-results` | Save final metrics to a CSV file in `results/` | `False` |

---

## 🐳 DOCKER DISTRIBUTED MODE (Phase 9)
Use these commands when you want to run the **true distributed version** where every worker is a separate container.

### 🏗️ BUILD COMMANDS

### Build Everything (All Services)
```bash
# Build and start all services (containers + images)
sudo docker-compose up -d --build

# Build only (without starting)
sudo docker-compose build
```

### Build Individual Services
```bash
# Build master only
sudo docker-compose build master
sudo docker-compose up -d master

# Build worker-0 only
sudo docker-compose build worker-0
sudo docker-compose up -d worker-0

# Build worker-1 only
sudo docker-compose build worker-1
sudo docker-compose up -d worker-1

# Build worker-2 only
sudo docker-compose build worker-2
sudo docker-compose up -d worker-2

# Build worker-3 only
sudo docker-compose build worker-3
sudo docker-compose up -d worker-3
```

### Build Workers One by One (To Save Disk Space)
```bash
sudo docker-compose up -d --build master
sudo docker-compose up -d --build worker-0
sudo docker-compose up -d --build worker-1
sudo docker-compose up -d --build worker-2
sudo docker-compose up -d --build worker-3
```

---

## 🚀 START COMMANDS

### Start Everything
```bash
# Start all services (if containers exist but stopped)
sudo docker-compose start

# Start all services (create containers if needed)
sudo docker-compose up -d
```

### Start Individual Services
```bash
sudo docker-compose start master
sudo docker-compose start worker-0
sudo docker-compose start worker-1
sudo docker-compose start worker-2
sudo docker-compose start worker-3
```

---

## 🛑 STOP COMMANDS

### Stop Everything
```bash
# Stop all containers (keep images for fast restart)
sudo docker-compose stop

# Stop containers AND remove them (but keep images)
sudo docker-compose down
```

### Stop Individual Service
```bash
sudo docker-compose stop master
sudo docker-compose stop worker-0
```

---

## 🔄 RESTART COMMANDS

### Restart Everything
```bash
# Restart all services
sudo docker-compose restart

# Stop and remove, then start fresh
sudo docker-compose down
sudo docker-compose up -d
```

### Restart Individual Service
```bash
sudo docker-compose restart master
sudo docker-compose restart worker-0
```

---

## 🗑️ DELETE/REMOVE COMMANDS

### Delete Containers Only (Keep Images)
```bash
# Remove all containers for this project
sudo docker-compose down

# Remove specific container
sudo docker rm distributed-llm-load-balancer-worker-0-1
```

### Delete Containers + Images (Full Clean)
```bash
# Remove everything (containers + images)
sudo docker-compose down --rmi all

# Remove everything + volumes (delete all data)
sudo docker-compose down --rmi all -v
```

### Delete Everything Docker-Related for This Project
```bash
# Stop everything
sudo docker-compose down

# Remove images
sudo docker rmi distributed-llm-load-balancer-master
sudo docker rmi distributed-llm-load-balancer-worker-0
sudo docker rmi distributed-llm-load-balancer-worker-1
sudo docker rmi distributed-llm-load-balancer-worker-2
sudo docker rmi distributed-llm-load-balancer-worker-3

# Remove volumes
sudo docker volume rm distributed-llm-load-balancer_chroma-data
```

### Delete All Unused Docker Resources (Global)
```bash
# Remove all stopped containers, unused networks, dangling images
sudo docker system prune

# Remove EVERYTHING unused (images, containers, volumes, cache)
sudo docker system prune -a --volumes
```

---

## 📊 CHECK SIZE & STATUS COMMANDS

### Check Container Status
```bash
# Show running containers
sudo docker-compose ps

# Show all containers (including stopped)
sudo docker-compose ps -a

# Show all Docker containers (from any project)
sudo docker ps -a
```

### Check Disk Usage
```bash
# Check system disk space
df -h /

# Check Docker disk usage
sudo docker system df

# Check Docker disk usage (detailed)
sudo docker system df -v

# Check size of specific images
sudo docker images | grep distributed-llm

# Check size of specific container
sudo docker ps -a --size
```

### Check Individual Container Size
```bash
# List containers with sizes
sudo docker ps -a --size --format "table {{.Names}}\t{{.Size}}\t{{.Status}}"

# Check specific container size
sudo docker inspect distributed-llm-load-balancer-master-1 | grep -i "size"
```

---

## 🔍 LOGS & MONITORING

### View Logs
```bash
# View all logs
sudo docker-compose logs

# View last 50 lines
sudo docker-compose logs --tail=50

# Follow logs in real-time
sudo docker-compose logs -f

# View specific service logs
sudo docker-compose logs master
sudo docker-compose logs worker-0 --tail=30 -f
```

---

## 🧪 TEST COMMANDS

### Test if System is Working
```bash
# Health check
curl http://localhost:8000/health

# Check workers
curl http://localhost:8000/workers/status

# Send test request
curl -X POST http://localhost:8000/request \
  -H "Content-Type: application/json" \
  -d '{"id": 1, "query": "What is distributed computing?"}'
```

---

## 📝 QUICK REFERENCE CARD

| Action | Command |
| :--- | :--- |
| **Build all** | `sudo docker-compose up -d --build` |
| **Start all** | `sudo docker-compose up -d` |
| **Stop all** | `sudo docker-compose down` |
| **Restart all** | `sudo docker-compose restart` |
| **Delete all (keep code)** | `sudo docker-compose down --rmi all -v` |
| **Check status** | `sudo docker-compose ps` |
| **Check all (stopped)** | `sudo docker-compose ps -a` |
| **Check disk** | `df -h /` |
| **Check Docker disk** | `sudo docker system df` |
| **View logs** | `sudo docker-compose logs -f` |

---

## 🚨 BAD COMMANDS (AVOID THESE UNLESS INTENTIONAL)
```bash
# ❌ Deletes ALL Docker images (including base images)
sudo docker image prune -a

# ❌ Deletes ALL unused volumes (can delete data)
sudo docker volume prune

# ❌ Deletes ALL build cache (slows future builds)
sudo docker builder prune -a
```

---

## 💡 BEST PRACTICES

### For Daily Work (Fastest)
```bash
# Stop for the day (keeps images)
sudo docker-compose down

# Start next day (instant)
sudo docker-compose up -d
```

### For Disk Space Cleanup (Without Losing Ability to Restart)
```bash
# Remove stopped containers only
sudo docker container prune

# Remove unused networks only
sudo docker network prune
```

### For Complete Fresh Start
```bash
# Nuclear option (only after project is done)
sudo docker-compose down --rmi all -v
sudo docker system prune -a --volumes
```
