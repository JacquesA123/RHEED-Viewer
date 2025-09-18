#!/usr/bin/env python
# pyrometer.py — Exactus Modbus RTU bring-up & control

from datetime import datetime, timezone
import argparse, time, struct, sys
from serial.tools import list_ports
print([p.device for p in list_ports.comports()])
from pymodbus.exceptions import ModbusIOException
from pymodbus.exceptions import ModbusException
try:
    from pymodbus.client import ModbusSerialClient
    from pymodbus.exceptions import ModbusException
except Exception as e:
    print("Missing pymodbus. Install with: py -m pip install pymodbus pyserial")
    raise

# ----------- Register map (from manual) -----------
# Measurements (float32 across two 16-bit regs; even=hi, odd=lo)
REG_CH1_TEMP = 0x0000
REG_CH1_CURR = 0x0004
REG_AMBIENT  = 0x0800

# Config & info (U16 unless noted)
REG_CFG1   = 0x1000
REG_CFG2   = 0x1001
REG_ADDR   = 0x1007
REG_BAUD   = 0x1008
REG_RATE   = 0x1011        # "Pyrometer Sample Rate" / TemperaSure "Graph Rate" (Hz)
REG_NAME0  = 0x1100        # 32 regs; 1 ASCII byte per reg; null-terminated
REG_VER    = 0x1300        # hi=major, lo=minor
REG_BUILD  = 0x1301
REG_SN0    = 0x1305        # 9 regs; 1 ASCII byte per reg

# Optional (if your firmware supports)
REG_CMD    = 0x8000
CMD_SAVE   = 0x7001        # save to EEPROM
CMD_REBOOT = 0x8080        # soft reboot

# ----------- Pymodbus compatibility wrappers -----------
def _read_holding(client, addr, count, device_id):
    for kw in ("device_id", "slave", "unit"):
        try: return client.read_holding_registers(addr, count, **{kw: device_id})
        except TypeError: pass
    return client.read_holding_registers(address=addr, count=count, device_id=device_id)

def _write_register(client, addr, value, device_id):
    for kw in ("device_id", "slave", "unit"):
        try: return client.write_register(addr, value, **{kw: device_id})
        except TypeError: pass
    return client.write_register(address=addr, value=value, device_id=device_id)

def _read_coils(client, addr, count, device_id):
    for kw in ("device_id", "slave", "unit"):
        try: return client.read_coils(addr, count, **{kw: device_id})
        except TypeError: pass
    return client.read_coils(address=addr, count=count, device_id=device_id)

# ----------- Conversions -----------
def regs_to_f32(hi, lo):
    raw = (hi << 16) | lo
    return struct.unpack('>f', raw.to_bytes(4, 'big'))[0]

def regs_to_ascii_byte_per_reg(regs):
    b = bytes(r & 0xFF for r in regs)
    return b.split(b'\x00', 1)[0].decode('ascii', errors='ignore')


def _resp_ok(rr, need):
    return hasattr(rr, "isError") and not rr.isError() and hasattr(rr, "registers") and len(rr.registers) >= need

def read_u16(client, addr, device_id):
    rr = _read_holding(client, addr, 1, device_id)
    if not _resp_ok(rr, 1):
        raise ModbusException(f"No reply / bad reply reading 0x{addr:04X}")
    return rr.registers[0]

def read_f32(client, addr, device_id, word_swap=False):
    rr = _read_holding(client, addr, 2, device_id)
    if not _resp_ok(rr, 2):
        raise ModbusException(f"No reply / bad reply reading 0x{addr:04X} (needs 2 regs)")
    hi, lo = rr.registers
    if word_swap:
        hi, lo = lo, hi
    return regs_to_f32(hi, lo)

def read_ascii(client, addr, n_regs, device_id):
    rr = _read_holding(client, addr, n_regs, device_id)
    if rr.isError(): raise ModbusException(rr)  # type: ignore
    return regs_to_ascii_byte_per_reg(rr.registers)

# ----------- Commands -----------
def cmd_probe(args):
    client = ModbusSerialClient(port=args.port, baudrate=args.baud, parity=args.parity,
                                stopbits=args.stopbits, bytesize=args.bytesize, timeout=args.timeout)
    if not client.connect():
        print("❌ Could not open serial port. Close TemperaSure or any serial monitor, and confirm the COM port.")
        return 2

    try:
        print(f"[{datetime.now().isoformat(timespec='seconds')}] Connected {args.port} @ {args.baud} 8{args.parity}{args.stopbits} (ID={args.id})")

        # Identity/config (best-effort)
        try:
            addr   = read_u16(client, REG_ADDR, args.id)
            baud   = read_u16(client, REG_BAUD, args.id)
            rate   = read_u16(client, REG_RATE, args.id)
            ver    = read_u16(client, REG_VER, args.id)
            build  = read_u16(client, REG_BUILD, args.id)
            name   = read_ascii(client, REG_NAME0, 32, args.id)
            serial = read_ascii(client, REG_SN0,  9,  args.id)
            print(f"Addr={addr}  BaudCode={baud}  GraphRate={rate} Hz  Ver={ver>>8}.{ver&0xFF}  Build={build>>8}.{build&0xFF}")
            print(f"Name='{name}'  Serial='{serial}'")
        except Exception as e:
            print("ℹ️ Identity read partial/failed:", e)

        # Measurements; try word order once if implausible
        def plausible_temp(x): return -100.0 < x < 3000.0
        word_swap = False
        try:
            t = read_f32(client, REG_CH1_TEMP, args.id, word_swap=False)
            i = read_f32(client, REG_CH1_CURR, args.id, word_swap=False)
            if not plausible_temp(t):
                t = read_f32(client, REG_CH1_TEMP, args.id, word_swap=True)
                i = read_f32(client, REG_CH1_CURR, args.id, word_swap=True)
                word_swap = True
            print(f"T={t:.3f} °C   I={i:.6g}   (word_swap={word_swap})")
        except Exception as e:
            print("ℹ️ Float read failed:", e)

        # Ambient (optional)
        try:
            amb = read_f32(client, REG_AMBIENT, args.id, word_swap=word_swap)
            print(f"Ambient={amb:.3f} °C")
        except Exception:
            pass

        print("✅ Probe complete.")
        return 0
    finally:
        client.close()
        print("🔌 Closed serial.")

def cmd_poll(args):
    client = ModbusSerialClient(port=args.port, baudrate=args.baud, parity=args.parity,
                                stopbits=args.stopbits, bytesize=args.bytesize, timeout=args.timeout)
    if not client.connect():
        print("❌ Could not open serial port.")
        return 2

    try:
        period = 1.0 / max(0.1, float(args.hz))
        t_end = time.time() + float(args.duration)
        while time.time() < t_end:
            try:
                t = read_f32(client, REG_CH1_TEMP, args.id)
                i = read_f32(client, REG_CH1_CURR, args.id)
                print(f"T={t:8.2f} °C   I={i:.6g}", end="\r", flush=True)
            except ModbusException as e:
                print("\nRead error:", e); break
            time.sleep(period)
        print()
        return 0
    finally:
        client.close()
        print("🔌 Closed serial.")

def cmd_set_rate(args):
    if not (1 <= args.rate <= 1000):
        print("Rate must be between 1 and 1000 Hz (adjust if your device supports more).")
        return 2

    client = ModbusSerialClient(port=args.port, baudrate=args.baud, parity=args.parity,
                                stopbits=args.stopbits, bytesize=args.bytesize, timeout=args.timeout)
    if not client.connect():
        print("❌ Could not open serial port.")
        return 2

    try:
        old = read_u16(client, REG_RATE, args.id)
        print(f"Current Graph/Sample Rate: {old} Hz")
        wr = _write_register(client, REG_RATE, int(args.rate), args.id)
        if wr.isError(): raise ModbusException(wr)  # type: ignore
        time.sleep(0.05)
        if args.persist:
            try:
                wr2 = _write_register(client, REG_CMD, CMD_SAVE, args.id)
                if hasattr(wr2, "isError") and wr2.isError():  # type: ignore
                    print("(EEPROM save command failed; device may auto-persist or use a different method.)")
                else:
                    print("Persisted to EEPROM.")
            except Exception as e:
                print("(EEPROM save not supported on this firmware? Skipping.)", e)
        confirmed = read_u16(client, REG_RATE, args.id)
        print(f"Confirmed rate: {confirmed} Hz")
        return 0
    finally:
        client.close()
        print("🔌 Closed serial.")




def cmd_scan(args):
    client = ModbusSerialClient(
        port=args.port, baudrate=args.baud,
        parity=args.parity, stopbits=args.stopbits,
        bytesize=args.bytesize, timeout=max(args.timeout, 1.5),
    )
    if not client.connect():
        print("❌ Could not open serial port."); return 2
    try:
        print(f"Scanning device IDs {args.id_min}..{args.id_max} at {args.baud} baud on {args.port}...")
        hits = []
        for sid in range(args.id_min, args.id_max + 1):
            try:
                rr = client.read_holding_registers(address=0x1300, count=1, device_id=sid)
                if hasattr(rr, "isError") and not rr.isError() and getattr(rr, "registers", None):
                    ver = rr.registers[0]
                    print(f"✅ device_id={sid}  version={ver>>8}.{ver&0xFF}")
                    hits.append(sid)
            except ModbusIOException:
                # no reply for this ID; keep scanning
                continue
            except Exception as e:
                print(f"(sid {sid}) {e}")
        if not hits:
            print("— no responders —")
        return 0
    finally:
        client.close(); print("🔌 Closed serial.")



# ----------- CLI -----------
def main(argv=None):
    p = argparse.ArgumentParser(prog="pyrometer", description="Exactus pyrometer Modbus RTU CLI")
    p.add_argument("--port", default="COM4")
    p.add_argument("--baud", type=int, default=115200)
    p.add_argument("--id",   type=int, default=1, help="Modbus device/slave ID")
    p.add_argument("--timeout", type=float, default=1.0)
    p.add_argument("--parity", default="N")
    p.add_argument("--stopbits", type=int, default=1)
    p.add_argument("--bytesize", type=int, default=8)
    sub = p.add_subparsers(dest="cmd", required=True)

    sub.add_parser("probe", help="Connect and read identity + basic measurements")
    sp = sub.add_parser("poll", help="Live values for a short time")
    sp.add_argument("--hz", type=float, default=2.0)
    sp.add_argument("--duration", type=float, default=10.0)

    sr = sub.add_parser("set-rate", help="Set device sample/graph rate (REG 0x1011)")
    sr.add_argument("--rate", type=int, required=True)
    sr.add_argument("--persist", action="store_true", help="Save to EEPROM if supported")

    sc = sub.add_parser("scan", help="Scan Modbus device IDs at the given baud")
    sc.add_argument("--id-min", type=int, default=1)
    sc.add_argument("--id-max", type=int, default=32)


    args = p.parse_args(argv)

    if args.cmd == "probe":     return cmd_probe(args)
    if args.cmd == "poll":      return cmd_poll(args)
    if args.cmd == "set-rate":  return cmd_set_rate(args)
    if args.cmd == "scan":      return cmd_scan(args)
    p.print_help()
    return 2

if __name__ == "__main__":
    raise SystemExit(main())

