document.addEventListener('DOMContentLoaded', () => {
    const gridEl = document.getElementById('grid-world');
    let currentTool = 1; // Default to Obstacle
    let isTraining = false;

    // Initialize Tool Selection
    document.querySelectorAll('.tool-btn').forEach(btn => {
        btn.addEventListener('click', (e) => {
            document.querySelectorAll('.tool-btn').forEach(b => b.classList.remove('active'));
            e.target.classList.add('active');
            currentTool = parseInt(e.target.dataset.tool);
        });
    });

    // Grid Initialization
    function renderGrid(grid, obstacles, start, goal) {
        gridEl.innerHTML = '';
        for (let r = 0; r < 10; r++) {
            for (let c = 0; c < 10; c++) {
                const cell = document.createElement('div');
                cell.classList.add('cell');
                cell.dataset.r = r;
                cell.dataset.c = c;

                // Determine class based on state
                // We use coordinates check for accuracy
                const isObs = obstacles.some(o => o[0] === r && o[1] === c);
                const isStart = start[0] === r && start[1] === c;
                const isGoal = goal[0] === r && goal[1] === c;

                if (isObs) cell.classList.add('obstacle');
                if (isStart) {
                    cell.classList.add('start');
                    cell.innerText = 'S';
                }
                if (isGoal) {
                    cell.classList.add('goal');
                    cell.innerText = 'G';
                }

                cell.addEventListener('click', () => handleCellClick(r, c));
                gridEl.appendChild(cell);
            }
        }
    }

    // API Calls
    async function refreshGrid() {
        const res = await fetch('/api/init');
        const data = await res.json();
        renderGrid(data.grid, data.obstacles, data.start, data.goal);
    }

    async function handleCellClick(r, c) {
        if (isTraining) return;

        await fetch('/api/set_cell', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ row: r, col: c, type: currentTool })
        });
        refreshGrid();
        resetResultsUI();
    }

    document.getElementById('random-btn').addEventListener('click', async () => {
        if (isTraining) return;
        await fetch('/api/randomize', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ num_obstacles: 15 })
        });
        refreshGrid();
        resetResultsUI();
    });

    document.getElementById('train-btn').addEventListener('click', async () => {
        if (isTraining) return;

        isTraining = true;
        updateStatus("正在重新训练新模型 (Reset)...", "#f39c12");

        try {
            const episodes = document.getElementById('episodes-input').value;
            const res = await fetch('/api/train', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ episodes: parseInt(episodes) })
            });
            const data = await res.json();

            if (data.error) {
                alert(data.error);
                updateStatus("出错啦", "red");
                return;
            }

            displayResults(data);
            updateStatus("训练完成 (新模型已就绪)", "#27ae60");

        } catch (e) {
            console.error(e);
            updateStatus("网络错误", "red");
        } finally {
            isTraining = false;
        }
    });

    document.getElementById('test-btn').addEventListener('click', async () => {
        if (isTraining) return;

        isTraining = true;
        updateStatus("正在使用现有模型测试 (泛化)...", "#e67e22");

        try {
            const res = await fetch('/api/test', { method: 'POST' });
            const data = await res.json();

            if (data.error) {
                alert(data.error);
                updateStatus("无法测试: " + data.error, "red");
                return;
            }

            displayResults(data);
            updateStatus("泛化测试完成", "#27ae60");

        } catch (e) {
            console.error(e);
            updateStatus("网络错误", "red");
        } finally {
            isTraining = false;
        }
    });

    function updateStatus(msg, color) {
        const el = document.getElementById('status-indicator');
        el.innerText = msg;
        if (color) el.style.color = color;
    }

    function resetResultsUI() {
        document.getElementById('output-gallery').style.display = 'none';
        document.getElementById('text-results').style.display = 'none';
        updateStatus("地图已变更 (模型保留, 可点击'验证模型')", "#34495e");
    }

    function displayResults(data) {
        // Text stats
        document.getElementById('text-results').style.display = 'block';

        if (data.is_unreachable) {
            document.getElementById('res-success').innerText = "无法到达 (Unreachable)";
            document.getElementById('res-steps').innerText = "-";
            document.getElementById('res-reward').innerText = "-";
            updateStatus("目标无法到达 (Unreachable)", "red");
            document.getElementById('output-gallery').style.display = 'none';
            return;
        }

        document.getElementById('res-success').innerText = data.success ? "是" : "否";
        document.getElementById('res-steps').innerText = data.steps;
        document.getElementById('res-reward').innerText = data.reward.toFixed(1);

        // Images
        document.getElementById('output-gallery').style.display = 'block';

        // Helper to update img src with timestamp to force refresh
        const updateImg = (id, filename) => {
            if (filename) {
                document.getElementById(id).src = `/static/images/${filename}`;
            }
        };

        if (data.images) {
            updateImg('img-path', data.images.path);
            updateImg('img-route-map', data.images.route_map);
            updateImg('img-curves', data.images.curves);
            updateImg('img-policy', data.images.policy);
            updateImg('img-value', data.images.value);
            updateImg('img-efficiency', data.images.efficiency);
        }
    }

    // Modal Logic
    const modal = document.getElementById('image-modal');
    const modalImg = document.getElementById("modal-img");
    const captionText = document.getElementById("caption");
    const span = document.getElementsByClassName("close")[0];
    const gallery = document.getElementById('output-gallery');

    // Use event delegation for better performance and reliability
    gallery.addEventListener('click', (e) => {
        if (e.target.classList.contains('zoomable')) {
            modal.style.display = "block";
            modalImg.src = e.target.src;
            captionText.innerHTML = e.target.alt;
        }
    });

    span.onclick = function () {
        modal.style.display = "none";
    }

    modal.onclick = function (event) {
        if (event.target === modal) {
            modal.style.display = "none";
        }
    }

    // Initial Load
    refreshGrid();
});
