'''
This library performs various calculations to analyse the OOS data.
'''

import numpy as np
from scipy.optimize import leastsq

class OOSAnonymisation: # pylint: disable=too-many-arguments, too-many-instance-attributes, too-many-locals
    """
    Processor for Out-Of-Straightness (OOS) pipeline survey and route data.
    Provides methods for cleaning, sectioning, and coordinate normalization.
    """
    def __init__(
            self,
            *,
            route_development,
            route_pipeline_group,
            route_pipeline,
            route_kp_from,
            route_kp_to,
            route_pipeline_section_type,
            route_curve_radius,
            survey_development,
            survey_type,
            survey_pipeline_group,
            survey_pipeline,
            survey_kp,
            survey_easting,
            survey_northing,
        ):
        """
        Initialize an OOSAnonymisation object with route and survey data.

        Parameters
        ----------
        route_development : array-like
            Development identifiers for each route section.
        route_pipeline_group : array-like
            Pipeline group identifiers for each route section.
        route_pipeline : array-like
            Pipeline identifiers for each route section.
        route_kp_from : array-like
            Start KP (kilometer point) for each route section.
        route_kp_to : array-like
            End KP (kilometer point) for each route section.
        route_pipeline_section_type : array-like
            Section type (e.g., 'Straight', 'Curve', etc.) for each route section.
        route_curve_radius : array-like
            Design curve radius for each route section.
        survey_development : array-like
            Development identifiers for each survey point.
        survey_type : array-like
            Survey type for each survey point.
        survey_pipeline_group : array-like
            Pipeline group identifiers for each survey point.
        survey_pipeline : array-like
            Pipeline identifiers for each survey point.
        survey_kp : array-like
            KP (kilometer point) for each survey point.
        survey_easting : array-like
            Easting coordinate for each survey point.
        survey_northing : array-like
            Northing coordinate for each survey point.

        Notes
        -----
        All input arrays are converted to NumPy arrays and stored as attributes. Additional
        attributes for section types, numbers, radii, and modified coordinates are
        initialized as NumPy arrays and will be populated by processing methods.
        """
        self.route_development = np.asarray(route_development, dtype=object)
        self.route_pipeline_group = np.asarray(route_pipeline_group, dtype=object)
        self.route_pipeline = np.asarray(route_pipeline, dtype=object)
        self.route_kp_from = np.asarray(route_kp_from, dtype=float)
        self.route_kp_to = np.asarray(route_kp_to, dtype=float)
        self.route_pipeline_section_type = np.asarray(route_pipeline_section_type, dtype=object)
        self.route_curve_radius = np.asarray(route_curve_radius, dtype=float)
        self.survey_development = np.asarray(survey_development, dtype=object)
        self.survey_type = np.asarray(survey_type, dtype=object)
        self.survey_pipeline_group = np.asarray(survey_pipeline_group, dtype=object)
        self.survey_pipeline = np.asarray(survey_pipeline, dtype=object)
        self.survey_kp = np.asarray(survey_kp, dtype=float)
        self.survey_easting = np.asarray(survey_easting, dtype=float)
        self.survey_northing = np.asarray(survey_northing, dtype=float)
        self.survey_section_type = np.full(
            self.survey_kp.shape[0], '', dtype=object
        )
        self.survey_section_no = np.full(
            self.survey_kp.shape[0], np.nan, dtype=float
        )
        self.survey_feature = np.full(
            self.survey_kp.shape[0], '', dtype=object
        )
        self.survey_design_route_curve_radius = np.full(
            self.survey_kp.shape[0], np.nan, dtype=float
        )
        self.survey_actual_route_curve_radius = np.full(
            self.survey_kp.shape[0], np.nan, dtype=float
        )
        self.survey_section_easting_mod = np.full(
            self.survey_kp.shape[0], np.nan, dtype=float
        )
        self.survey_section_northing_mod = np.full(
            self.survey_kp.shape[0], np.nan, dtype=float
        )
        self.survey_section_kp_mod = np.full(
            self.survey_kp.shape[0], np.nan, dtype=float
        )

    def process(self, anonymise: bool = False):
        """
        Run the OOS processing pipeline.

        Parameters
        ----------
        anonymise : bool, default False
            If True, runs the full anonymisation pipeline (with cleaning and grouping).
            If False, runs only the basic section processing.
        """
        self._add_pipeline_section()
        self._add_feature_labels()
        self._add_design_route_curve_radius()
        self._add_actual_route_curve_radius()
        self._transform_pipeline_section_coordinates()
        if anonymise:
            self._trim_feature_kps()
            self._trim_straight_section_ends()
            self._add_pipeline_group_section()
            self._transform_pipeline_group_section_coordinates()

    def _add_pipeline_section(self):
        """
        Add section type ('Straight' or 'Curve') and section number to each survey point.

        For each survey point, determines if it falls within a non-straight route section (based on
        matching pipeline group, pipeline, and KP range) and adds 'Curve' or 'Straight' to
        self.survey_section_type. Then, adds a unique section number (self.survey_section_no) to
        each continuous section of the same type within each (pipeline group, pipeline, survey type)
        group.

        Updates
        -------
        self.survey_section_type : np.ndarray
            Section type ('Straight' or 'Curve') for each survey point.
        self.survey_section_no : np.ndarray
            Section number for each survey point, incremented for each new section.

        Returns
        -------
        None
        """
        # Build mask for non-straight route sections
        mask = ~np.isin(self.route_pipeline_section_type, ['LBMD', 'PWMD', 'ILT', 'Straight'])
        # Indices of route sections that are not straight
        idxs = np.where(mask)[0]
        mask_curve = np.full(self.survey_kp.shape[0], False)
        for i in idxs:
            cond = (
                (self.survey_development == self.route_development[i]) &
                (self.survey_pipeline_group == self.route_pipeline_group[i]) &
                (self.survey_pipeline == self.route_pipeline[i]) &
                (self.survey_kp >= self.route_kp_from[i]) &
                (self.survey_kp <= self.route_kp_to[i])
            )
            mask_curve = mask_curve | cond
        # Create or update section type array
        section_type = np.full(self.survey_kp.shape[0], 'Straight', dtype=object)
        section_type[mask_curve] = 'Curve'
        self.survey_section_type = section_type

        # Add Section No for each continuous section
        group_keys = [self.survey_development, self.survey_pipeline_group, self.survey_pipeline]
        section_no = np.zeros(self.survey_kp.shape[0], dtype=int)
        prev_keys = None
        prev_type = None
        current_section = 0
        for i in range(self.survey_kp.shape[0]):
            keys = tuple(key[i] for key in group_keys)
            curr_type = section_type[i]
            if keys != prev_keys:
                current_section = 1
            elif curr_type != prev_type:
                current_section += 1
            section_no[i] = current_section
            prev_keys = keys
            prev_type = curr_type
        self.survey_section_no = section_no

    def _add_feature_labels(self):
        """
        Adds a new attribute 'survey_feature' to the survey points.
        For each feature (PWMD, LBMD, ILT) in the route, finds the closest survey point
        (matching development, pipeline group, pipeline) in KP and adds the feature name
        at that position.
        All other survey points have an empty string.
        """
        feature_types = ['PWMD', 'LBMD', 'ILT']
        for i, section_type in enumerate(self.route_pipeline_section_type):
            if section_type in feature_types:
                dev = self.route_development[i]
                group = self.route_pipeline_group[i]
                pipe = self.route_pipeline[i]
                kp = self.route_kp_from[i]
                # Find matching survey points
                mask = (
                    (self.survey_development == dev) &
                    (self.survey_pipeline_group == group) &
                    (self.survey_pipeline == pipe)
                )
                if not np.any(mask):
                    continue
                kp_survey = self.survey_kp[mask]
                idx_in_mask = np.argmin(np.abs(kp_survey - kp))
                idx = np.where(mask)[0][idx_in_mask]
                self.survey_feature[idx] = section_type

    def _add_design_route_curve_radius(self):
        """
        Add the 'Design Route Curve Radius' from route data to each survey point where the
        section type is 'Curve'.

        For each survey point with section type 'Curve', finds the matching route section (by
        development, pipeline group, pipeline, and KP within [KP From, KP To]) and adds
        the corresponding route curve radius to self.survey_design_route_curve_radius.

        Updates
        -------
        self.survey_design_route_curve_radius : np.ndarray
            Design route curve radii for each survey point (NaN for non-curve points).

        Returns
        -------
        None
        """
        mask_curve = self.survey_section_type == 'Curve'
        for i, _ in enumerate(self.route_development):
            cond = (
                (self.survey_development == self.route_development[i]) &
                (self.survey_pipeline_group == self.route_pipeline_group[i]) &
                (self.survey_pipeline == self.route_pipeline[i]) &
                (self.survey_kp >= self.route_kp_from[i]) &
                (self.survey_kp <= self.route_kp_to[i]) &
                mask_curve
            )
            self.survey_design_route_curve_radius[cond] = self.route_curve_radius[i]

    def _add_actual_route_curve_radius(self):
        """
        Fit a circle to each group of survey points where the section type is 'Curve' and add
        the fitted radius.

        For each group of survey points (grouped by development, pipeline group, pipeline, survey
        type, and section number) where all points are of type 'Curve' and there are more than 3
        points, fit a circle to the (easting, northing) coordinates and add the resulting radius
        to self.survey_actual_route_curve_radius for those points. Non-curve points or groups with
        insufficient points remain NaN.

        Updates
        -------
        self.survey_actual_route_curve_radius : np.ndarray
            Fitted curve radii for each survey point (NaN for non-curve points or insufficient
            data).

        Returns
        -------
        None
        """
        def calc_radius(x, y, xc, yc):
            return np.sqrt((x - xc) ** 2 + (y - yc) ** 2)
        def fit_circle(x, y):
            x_m = np.mean(x)
            y_m = np.mean(y)
            def f(c):
                radius_i = calc_radius(x, y, *c)
                return radius_i - radius_i.mean()
            center_estimate = x_m, y_m
            center, _ = leastsq(f, center_estimate)
            xc, yc = center
            radius_i = calc_radius(x, y, xc, yc)
            radius = radius_i.mean()
            return radius
        # Group by (development, survey_type, survey_pipeline_group, survey_pipeline, survey_section_no)
        group_tuples = list(zip(
            self.survey_development,
            self.survey_type,
            self.survey_pipeline_group,
            self.survey_pipeline,
            self.survey_section_no
        ))
        unique_groups = set(group_tuples)
        for group in unique_groups:
            mask = (
                (self.survey_development == group[0]) &
                (self.survey_type == group[1]) &
                (self.survey_pipeline_group == group[2]) &
                (self.survey_pipeline == group[3]) &
                (self.survey_section_no == group[4])
            )
            if np.all(self.survey_section_type[mask] == 'Curve') and np.sum(mask) > 3:
                x = self.survey_easting[mask]
                y = self.survey_northing[mask]
                radius = fit_circle(x, y)
                self.survey_actual_route_curve_radius[mask] = radius

    def _transform_pipeline_section_coordinates(self):
        """
        Compute and add section-modified coordinates and KP for each pipeline section.

        For each group of survey points (grouped by development, pipeline group, pipeline,
        survey type, and section number):
        - Shift Easting/Northing so the section starts at (0,0)
        - Rotate the section so it ends at (d, 0) (horizontal line)
        - Compute a section-modified KP (distance along the section, starting at 0)
        The results are stored in self.survey_section_easting_mod,
        self.survey_section_northing_mod, and self.survey_section_kp_mod for each survey point.

        Updates
        -------
        self.survey_section_easting_mod : np.ndarray
            Section-modified easting coordinates for each survey point.
        self.survey_section_northing_mod : np.ndarray
            Section-modified northing coordinates for each survey point.
        self.survey_section_kp_mod : np.ndarray
            Section-modified KP (distance along section, starting at 0) for each survey point.

        Returns
        -------
        None
        """

        group_tuples = list(zip(
            self.survey_development,
            self.survey_type,
            self.survey_pipeline_group,
            self.survey_pipeline,
            self.survey_section_no
        ))
        unique_groups = set(group_tuples)
        for group in unique_groups:
            mask = (
                (self.survey_development == group[0]) &
                (self.survey_type == group[1]) &
                (self.survey_pipeline_group == group[2]) &
                (self.survey_pipeline == group[3]) &
                (self.survey_section_no == group[4])
            )
            idx = np.where(mask)[0]
            if len(idx) == 0:
                continue
            x = self.survey_easting[idx]
            y = self.survey_northing[idx]
            x0, y0 = x[0], y[0]
            x_shift = x - x0
            y_shift = y - y0
            dx = x_shift[-1]
            dy = y_shift[-1]
            theta = -np.arctan2(dy, dx)
            x_mod = x_shift * np.cos(theta) - y_shift * np.sin(theta)
            y_mod = x_shift * np.sin(theta) + y_shift * np.cos(theta)
            kp_mod = np.append(0, np.cumsum(np.diff(self.survey_kp[idx])))
            self.survey_section_easting_mod[idx] = x_mod
            self.survey_section_northing_mod[idx] = y_mod
            self.survey_section_kp_mod[idx] = kp_mod

    def _trim_feature_kps(self):
        """
        Remove survey points where KP is within ±0.2 km of any KP From in the route data for
        matching development, pipeline group, and pipeline.

        Applies a mask to all survey arrays, shortening them in place. This is typically used to
        remove points near mitigation features (e.g., LBMD, PWMD, ILT) from the survey data.

        Updates
        -------
        All survey arrays are filtered in place, reducing their length.

        Returns
        -------
        None
        """
        mitigation_mask = np.isin(self.route_pipeline_section_type, ['LBMD', 'PWMD', 'ILT'])
        keep_mask = np.ones(self.survey_kp.shape[0], dtype=bool)
        for i in np.where(mitigation_mask)[0]:
            cond = (
                (self.survey_development == self.route_development[i]) &
                (self.survey_pipeline_group == self.route_pipeline_group[i]) &
                (self.survey_pipeline == self.route_pipeline[i]) &
                (np.abs(self.survey_kp - self.route_kp_from[i]) <= 0.2)
            )
            keep_mask = keep_mask & ~cond
        # Apply mask to all survey arrays
        self.survey_development = self.survey_development[keep_mask]
        self.survey_type = self.survey_type[keep_mask]
        self.survey_pipeline_group = self.survey_pipeline_group[keep_mask]
        self.survey_pipeline = self.survey_pipeline[keep_mask]
        self.survey_kp = self.survey_kp[keep_mask]
        self.survey_easting = self.survey_easting[keep_mask]
        self.survey_northing = self.survey_northing[keep_mask]
        self.survey_section_type = self.survey_section_type[keep_mask]
        self.survey_section_no = self.survey_section_no[keep_mask]
        self.survey_section_kp_mod = self.survey_section_kp_mod[keep_mask]
        self.survey_section_easting_mod = self.survey_section_easting_mod[keep_mask]
        self.survey_section_northing_mod = self.survey_section_northing_mod[keep_mask]
        self.survey_design_route_curve_radius = self.survey_design_route_curve_radius[keep_mask]
        self.survey_actual_route_curve_radius = self.survey_actual_route_curve_radius[keep_mask]
        self.survey_feature = self.survey_feature[keep_mask]

    def _trim_straight_section_ends(self):
        """
        Trim 50 meters from both ends of all Straight sections, but only when the section changes
        from Curve to Straight or from Straight to Curve.

        Applies a mask to all survey arrays, shortening them in place. Only points at the ends of
        straight sections adjacent to curves are removed.

        Updates
        -------
        All survey arrays are filtered in place, reducing their length.

        Returns
        -------
        None
        """
        n = self.survey_kp.shape[0]
        keep_mask = np.ones(n, dtype=bool)
        # Group by (Development, Pipeline Group, Pipeline, Survey Type, Section No)
        group_tuples = list(zip(
            self.survey_development,
            self.survey_type,
            self.survey_pipeline_group,
            self.survey_pipeline,
            self.survey_section_no
        ))
        unique_groups = set(group_tuples)
        for group in unique_groups:
            mask = (
                (self.survey_development == group[0]) &
                (self.survey_type == group[1]) &
                (self.survey_pipeline_group == group[2]) &
                (self.survey_pipeline == group[3]) &
                (self.survey_section_no == group[4])
            )
            idx = np.where(mask)[0]
            if len(idx) == 0:
                continue
            if not np.all(self.survey_section_type[idx] == 'Straight'):
                continue
            # Find previous and next section types
            prev_idx = idx[0] - 1 if idx[0] > 0 else None
            next_idx = idx[-1] + 1 if idx[-1] < n - 1 else None
            prev_type = self.survey_section_type[prev_idx] if prev_idx is not None else None
            next_type = self.survey_section_type[next_idx] if next_idx is not None else None
            # Only trim if adjacent to a Curve
            if prev_type == 'Curve':
                kp0 = self.survey_kp[idx].min()
                trim_mask = (self.survey_kp[idx] - kp0) < 0.05
                keep_mask[idx[trim_mask]] = False
            if next_type == 'Curve':
                kp1 = self.survey_kp[idx].max()
                trim_mask = (kp1 - self.survey_kp[idx]) < 0.05
                keep_mask[idx[trim_mask]] = False
        # Apply mask to all survey arrays
        self.survey_development = self.survey_development[keep_mask]
        self.survey_type = self.survey_type[keep_mask]
        self.survey_pipeline_group = self.survey_pipeline_group[keep_mask]
        self.survey_pipeline = self.survey_pipeline[keep_mask]
        self.survey_kp = self.survey_kp[keep_mask]
        self.survey_easting = self.survey_easting[keep_mask]
        self.survey_northing = self.survey_northing[keep_mask]
        self.survey_section_type = self.survey_section_type[keep_mask]
        self.survey_section_no = self.survey_section_no[keep_mask]
        self.survey_section_kp_mod = self.survey_section_kp_mod[keep_mask]
        self.survey_section_easting_mod = self.survey_section_easting_mod[keep_mask]
        self.survey_section_northing_mod = self.survey_section_northing_mod[keep_mask]
        self.survey_design_route_curve_radius = self.survey_design_route_curve_radius[keep_mask]
        self.survey_actual_route_curve_radius = self.survey_actual_route_curve_radius[keep_mask]
        self.survey_feature = self.survey_feature[keep_mask]

    def _add_pipeline_group_section(self):
        """
        Adds group-level section type ('Straight' or 'Curve') and section number to each survey
        point.

        For each survey point, adds the group section type (same as section type) and a unique
        group section number within each (development, pipeline group, survey type) group. The
        section number increments for each new continuous section of the same type.

        Updates
        -------
        self.survey_group_section_type : np.ndarray
            Group section type for each survey point.
        self.survey_group_section_no : np.ndarray
            Group section number for each survey point.

        Returns
        -------
        None
        """
        def remove_outliers_iqr(arr, whisker=1.5):
            q1 = np.nanpercentile(arr, 25)
            q3 = np.nanpercentile(arr, 75)
            iqr = q3 - q1
            lower = q1 - whisker * iqr
            upper = q3 + whisker * iqr
            mask = (arr >= lower) & (arr <= upper)
            return arr[mask]
        # Add group section number
        current_section = 0
        group_keys1 = [self.survey_development, self.survey_type, self.survey_pipeline_group]
        group_keys2 = group_keys1 + [self.survey_pipeline, self.survey_section_no]
        prev_keys1 = None
        prev_keys2 = None
        prev_kp = None
        # Group no. loop
        n = self.survey_kp.shape[0]
        group_section_no = np.zeros(n, dtype=int)
        kp_diff = np.diff(self.survey_kp)
        kp_diff_outliers = remove_outliers_iqr(kp_diff)
        avg_kp_diff = np.nanmean(kp_diff_outliers)
        for i in range(n):
            keys1 = tuple(key[i] for key in group_keys1)
            keys2 = tuple(key[i] for key in group_keys2)
            curr_kp = self.survey_kp[i]
            if keys1 != prev_keys1:
                current_section = 1  # Reset to 1 when group_keys changes
            elif keys2 != prev_keys2:
                current_section += 1  # Increase if section type changes
            elif prev_kp is not None and curr_kp - prev_kp > 1.5*avg_kp_diff:
                current_section += 1
            group_section_no[i] = current_section
            prev_keys1 = keys1
            prev_keys2 = keys2
            prev_kp = curr_kp
        self.survey_group_section_no = group_section_no
        self.survey_group_section_type = self.survey_section_type.copy()

    def _transform_pipeline_group_section_coordinates(self):
        """
        Compute and add group section-modified coordinates and KP for each pipeline group
        section.

        For each group of survey points (grouped by development, pipeline group, survey type,
        group section type, and group section number):
        - Shift Easting/Northing so the section starts at (0,0)
        - Rotate the section so it ends at (d, 0) (horizontal line)
        - Compute a group section-modified KP (distance along the section, starting at 0)
        The results are stored in self.survey_group_section_easting_mod,
        self.survey_group_section_northing_mod, and self.survey_group_section_kp_mod for each
        survey point.

        Updates
        -------
        self.survey_group_section_easting_mod : np.ndarray
            Group section-modified easting coordinates for each survey point.
        self.survey_group_section_northing_mod : np.ndarray
            Group section-modified northing coordinates for each survey point.
        self.survey_group_section_kp_mod : np.ndarray
            Group section-modified KP (distance along section, in km) for each survey point.

        Returns
        -------
        None
        """
        n = self.survey_kp.shape[0]
        self.survey_group_section_easting_mod = np.full(n, np.nan, dtype=float)
        self.survey_group_section_northing_mod = np.full(n, np.nan, dtype=float)
        self.survey_group_section_kp_mod = np.full(n, np.nan, dtype=float)
        # Group by (Development, Pipeline Group, Survey Type, Group Section Type)
        group_tuples = list(zip(
            self.survey_development,
            self.survey_type,
            self.survey_pipeline_group,
            self.survey_group_section_type
        ))
        unique_groups = set(group_tuples)
        for group in unique_groups:
            mask = (
                (self.survey_development == group[0]) &
                (self.survey_type == group[1]) &
                (self.survey_pipeline_group == group[2]) &
                (self.survey_group_section_type == group[3])
            )
            idx = np.where(mask)[0]

            # Sort idx by group section no to ensure correct order
            section_nos = self.survey_group_section_no[idx]
            unique_section_nos = np.unique(section_nos)
            shift_x = 0.0
            shift_y = 0.0
            for index, sec_no in enumerate(unique_section_nos):
                sec_mask = idx[section_nos == sec_no]
                x = self.survey_section_easting_mod[sec_mask]
                y = self.survey_section_northing_mod[sec_mask]

                # Compute average slope (angle) for the first 20 and last 20 points
                n_points = len(x)
                n_avg = min(20, n_points // 2)  # Use up to 20, but not more than half the section

                # If not the first section, align to average direction between last N points of
                # previous section and first N of current using robust vector method
                if index == 0:
                    theta = 0.0
                else:
                    prev_idx = np.where(section_nos == prev_sec_no)[0]
                    # Last n_avg points of previous section
                    x_prev = x_prev_all[-prev_n_avg:]
                    y_prev = y_prev_all[-prev_n_avg:]
                    # First n_avg points of current section
                    x_curr = x[:n_avg]
                    y_curr = y[:n_avg]
                    # Best linear fit for previous section
                    coeffs_prev = np.polyfit(x_prev, y_prev, 1)
                    slope_prev = coeffs_prev[0]
                    v1 = np.array([1.0, slope_prev])  # direction vector from slope
                    v1_u = v1 / np.linalg.norm(v1)
                    # Best linear fit for current section
                    coeffs_curr = np.polyfit(x_curr, y_curr, 1)
                    slope_curr = coeffs_curr[0]
                    v2 = np.array([1.0, slope_curr])
                    v2_u = v2 / np.linalg.norm(v2)
                    # Angle between the two vectors
                    dot = np.dot(v1_u, v2_u)
                    det = v1_u[0]*v2_u[1] - v1_u[1]*v2_u[0]
                    theta = -np.arctan2(det, dot)
                # Previous section
                prev_sec_no = sec_no
                prev_n_avg = n_avg
                # Shift to origin (0,0)
                x_to_origin = x - x[0]
                y_to_origin = y - y[0]
                # Apply rotation
                x_rot = x_to_origin * np.cos(theta) - y_to_origin * np.sin(theta)
                y_rot = x_to_origin * np.sin(theta) + y_to_origin * np.cos(theta)
                # Save previous
                x_prev_all = x_rot
                y_prev_all = y_rot
                # Apply shift for continuity
                x_mod = x_rot + shift_x
                y_mod = y_rot + shift_y
                # Define variables
                kp_mod = np.insert(np.cumsum(np.diff(self.survey_kp[sec_mask])), 0, 0)
                self.survey_group_section_easting_mod[sec_mask] = x_mod
                self.survey_group_section_northing_mod[sec_mask] = y_mod
                self.survey_group_section_kp_mod[sec_mask] = kp_mod
                # Prepare shift for next section: last point of this section
                shift_x = x_mod[-1]
                shift_y = y_mod[-1]

        # Align all sections to the x-axis
        for group in unique_groups:
            mask = (
                (self.survey_development == group[0]) &
                (self.survey_type == group[1]) &
                (self.survey_pipeline_group == group[2]) &
                (self.survey_group_section_type == group[3])
            )
            idx = np.where(mask)[0]

            # Use the group section modified coordinates
            x = self.survey_group_section_easting_mod[idx]
            y = self.survey_group_section_northing_mod[idx]

            # Compute angle to rotate so last point is on the x-axis
            dx = x[-1]
            dy = y[-1]
            theta = -np.arctan2(dy, dx)

            # Rotate all points
            x_aligned = x * np.cos(theta) - y * np.sin(theta)
            y_aligned = x * np.sin(theta) + y * np.cos(theta)

            # Store the aligned coordinates back
            self.survey_group_section_easting_mod[idx] = x_aligned
            self.survey_group_section_northing_mod[idx] = y_aligned

            # Ensure the max magnitude of northing is positive
            max_north = np.max(self.survey_group_section_northing_mod[idx])
            min_north = np.min(self.survey_group_section_northing_mod[idx])
            if abs(max_north) < abs(min_north):
                # Flip the sign of northing
                self.survey_group_section_northing_mod[idx] = -self.survey_group_section_northing_mod[idx]
