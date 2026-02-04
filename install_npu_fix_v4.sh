#!/bin/bash
# NPU Second Inference Fix v4 - Destroys HW command pool before CREATE
set -e

DRIVER_SRC="/usr/src/aipu-5.11.0"
DRIVER_FILE="$DRIVER_SRC/armchina-npu/aipu_job_manager.c"
BACKUP_FILE="$DRIVER_FILE.orig"

echo "=== NPU Fix v4 Installer ==="

if [ "$EUID" -ne 0 ]; then
    echo "Please run as root: sudo $0"
    exit 1
fi

if [ -f "$BACKUP_FILE" ]; then
    echo "[1/5] Restoring original source..."
    cp "$BACKUP_FILE" "$DRIVER_FILE"
else
    echo "[1/5] Creating backup..."
    cp "$DRIVER_FILE" "$BACKUP_FILE"
fi

echo "[2/5] Applying patch v4..."

python3 << 'PYSCRIPT'
driver_file = "/usr/src/aipu-5.11.0/armchina-npu/aipu_job_manager.c"

with open(driver_file, 'r') as f:
    content = f.read()

# Patch 1: Add variable declarations
old_decl = 'struct qos *qlist = NULL;\n\tstruct aipu_hold_tcb_buf *htbuf = NULL;'
new_decl = '''struct qos *qlist = NULL;
\tstruct aipu_job *check_job = NULL;
\tbool has_running_job = false;
\tstruct aipu_hold_tcb_buf *htbuf = NULL;'''

if old_decl not in content:
    print("ERROR: Could not find declaration block")
    exit(1)
content = content.replace(old_decl, new_decl)

# Patch 2: Replace trigger_type logic - destroy HW pool if stale
old_logic = '''\tif (!pool->created)
\t\ttrigger_type = ZHOUYI_TRIGGER_TYPE_CREATE;
\telse if (pool->aborted || !qlist->pool_head)
\t\ttrigger_type = ZHOUYI_TRIGGER_TYPE_UPDATE_DISPATCH;
\telse
\t\ttrigger_type = ZHOUYI_TRIGGER_TYPE_DISPATCH;

\tcheck_enable_tec_interrupts(manager, job);'''

new_logic = '''\t/*
\t * Fix v4 for second inference hang: If all previous jobs completed but
\t * the command pool wasn't destroyed yet (race condition), we must
\t * destroy the hardware command pool before creating a new one.
\t */
\thas_running_job = false;
\tlist_for_each_entry(check_job, &manager->scheduled_head->node, node) {
\t\tif (check_job->state == AIPU_JOB_STATE_RUNNING) {
\t\t\thas_running_job = true;
\t\t\tbreak;
\t\t}
\t}

\tif (!pool->created) {
\t\ttrigger_type = ZHOUYI_TRIGGER_TYPE_CREATE;
\t} else if (!has_running_job && qlist->pool_head) {
\t\t/*
\t\t * Stale pool detected: all jobs completed but pool still exists.
\t\t * Destroy the hardware command pool and reset software state.
\t\t */
\t\tpartition->ops->destroy_command_pool(partition, 0);
\t\tmemset(pool->qlist, 0, sizeof(*pool->qlist) * AIPU_JOB_QOS_MAX);
\t\tpool->created = false;
\t\tmanager->tec_intr_en = false;
\t\taipu_mm_set_final_htbuf_index(manager->mm, -1);
\t\t/* Re-select qlist after clearing */
\t\tif (job->desc.exec_flag & AIPU_JOB_EXEC_FLAG_QOS_SLOW)
\t\t\tqlist = &pool->qlist[AIPU_JOB_QOS_SLOW];
\t\telse
\t\t\tqlist = &pool->qlist[AIPU_JOB_QOS_FAST];
\t\ttrigger_type = ZHOUYI_TRIGGER_TYPE_CREATE;
\t} else if (pool->aborted || !qlist->pool_head) {
\t\ttrigger_type = ZHOUYI_TRIGGER_TYPE_UPDATE_DISPATCH;
\t} else {
\t\ttrigger_type = ZHOUYI_TRIGGER_TYPE_DISPATCH;
\t}

\tcheck_enable_tec_interrupts(manager, job);'''

if old_logic not in content:
    print("ERROR: Could not find trigger_type logic")
    exit(1)
content = content.replace(old_logic, new_logic)

with open(driver_file, 'w') as f:
    f.write(content)
print("      Patch v4 applied!")
PYSCRIPT

echo "[3/5] Removing old DKMS module..."
dkms remove aipu/5.11.0 --all 2>/dev/null || true

echo "[4/5] Rebuilding kernel module..."
cd "$DRIVER_SRC"
dkms add .
dkms build aipu/5.11.0
dkms install aipu/5.11.0

echo "[5/5] Reloading kernel module..."
rmmod aipu 2>/dev/null || true
modprobe aipu
echo "on" > /sys/devices/platform/CIXH4000:00/power/control 2>/dev/null || true

echo ""
echo "=== Done! ==="
