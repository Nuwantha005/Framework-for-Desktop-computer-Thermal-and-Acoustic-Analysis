import re

with open('src/solvers/actuator/coupling.py', 'r') as f:
    code = f.read()

# Replace system_q logic with disk.system_q
new_code = re.sub(
r"        for iteration in range\(1, max_iterations \+ 1\):.*?if active_residuals and all\(residual <= 1e-5 for residual in active_residuals\):",
r'''        for iteration in range(1, max_iterations + 1):
            # Total intake flow rate is the sum of system_q for all intake fans.
            # We assume fans with normal pointing in -y or -z or -x into the casing are intakes.
            # Or simpler: just sum the flows of all fans that pull air from outside.
            # Let's just sum all disk.system_q that have "intake" in their name, or if not found, use the first disk's q.
            intake_q = sum(d.system_q for d in self._disks if "intake" in d.config.name.lower())
            if intake_q == 0 and self._disks:
                intake_q = self._disks[0].system_q
                
            total_inlet_area = sum(inlet.area for inlet in self._inlets)
            for inlet in self._inlets:
                if inlet.area > 0 and total_inlet_area > 0:
                    # distribute total intake_q proportionally by area
                    inlet.source_strength = np.full(inlet.mesh.num_panels, 2.0 * intake_q / total_inlet_area)
            
            total_outlet_area = sum(outlet.area for outlet in self._outlets)
            for outlet in self._outlets:
                if outlet.area > 0 and total_outlet_area > 0:
                    outlet.source_strength = np.full(outlet.mesh.num_panels, -2.0 * intake_q / total_outlet_area)
                    
            for disk in self._disks:
                dp_curve = disk.curve.pressure_at(disk.system_q)
                disk.pressure_rise = dp_curve
                self._update_disk_doublet_strength(disk)

            disturbance = self._compute_body_normal_disturbance()
            self._body_solver = self._create_body_solver()
            self._body_solver.solve(normal_velocity_disturbance=disturbance)

            residuals = []
            bounds_reached = False
            
            for disk in self._disks:
                disk_velocity = self._velocity_at_disk(disk)
                disk.normal_velocity = compute_disk_normal_velocity(disk_velocity, disk.mesh)
                measured_q = integrate_flow_rate(disk.normal_velocity, disk.mesh)
                disk.flow_rate = measured_q
                
                if not disk.curve.contains_flow_rate(disk.system_q):
                    bounds_reached = True
                    warning = (
                        f"Fan '{disk.config.name}' flow rate {disk.system_q:.6e} m^3/s "
                        f"left fan-curve range [{disk.curve.q_min:.6e}, "
                        f"{disk.curve.q_max:.6e}] m^3/s; stopping ADM iteration."
                    )
                    self._warnings[disk.config.name] = warning
                    print(f"[ADM WARNING] {warning}")
                    
                dp_curve = disk.curve.pressure_at(disk.system_q)
                residual = measured_q - disk.system_q
                residuals.append(abs(residual))
                self._history.append(
                    ADMIterationRecord(
                        iteration=iteration,
                        disk_name=disk.config.name,
                        flow_rate=measured_q,
                        pressure_rise=dp_curve,
                        pressure_rise_curve=dp_curve,
                        pressure_residual=residual,
                    )
                )
                print(
                    f"[ADM] iter={iteration:03d} fan={disk.config.name} "
                    f"Q_sys={disk.system_q:.6e} m^3/s Q_meas={measured_q:.6e} m^3/s "
                    f"dp={dp_curve:.6e} Pa residual={residual:.6e} m^3/s"
                )

                if not bounds_reached and iteration < disk.config.max_iterations:
                    disk.system_q += disk.config.relaxation * residual

            if bounds_reached:
                break

            active_residuals = [
                residual
                for i, residual in enumerate(residuals)
                if iteration <= self._disks[i].config.max_iterations
            ]
            
            if active_residuals and all(residual <= 1e-5 for residual in active_residuals):''',
code, flags=re.DOTALL)

with open('src/solvers/actuator/coupling.py', 'w') as f:
    f.write(new_code)
