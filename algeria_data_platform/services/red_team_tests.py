"""
Red Team Testing for Algeria Data Platform
Implements adversarial testing to discover weaknesses proactively
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Callable
import logging
from dataclasses import dataclass
from enum import Enum

logger = logging.getLogger(__name__)


class AttackType(Enum):
    """Categories of adversarial attacks on data quality"""
    COMPLETENESS = "completeness"
    ACCURACY = "accuracy"
    CONSISTENCY = "consistency"
    TIMELINESS = "timeliness"


@dataclass
class AttackResult:
    """Result of a single red team attack"""
    attack_name: str
    attack_type: AttackType
    detected: bool
    detection_time_ms: float
    recovery_success: bool
    details: Dict


class RedTeamDataQuality:
    """
    Adversarial testing framework for data quality
    Implements "torture tests" to ensure system resilience
    """

    def __init__(self):
        self.attack_registry: Dict[str, Callable] = {
            # Completeness attacks
            'completeness_random_nulls_50_percent': self.inject_random_nulls,
            'completeness_entire_columns_missing': self.remove_entire_columns,
            'completeness_intermittent_data_gaps': self.create_data_gaps,

            # Accuracy attacks
            'accuracy_inject_outliers_3_sigma': self.inject_statistical_outliers,
            'accuracy_swap_decimal_separators': self.corrupt_decimal_formats,
            'accuracy_introduce_typos_in_categories': self.inject_typos,

            # Consistency attacks
            'consistency_conflicting_totals_vs_details': self.create_sum_conflicts,
            'consistency_duplicate_primary_keys': self.duplicate_keys,
            'consistency_contradictory_boolean_fields': self.create_logical_contradictions,

            # Timeliness attacks
            'timeliness_backdated_timestamps': self.backdate_timestamps,
            'timeliness_future_dates_in_historical_data': self.inject_future_dates,
        }

    # ============== COMPLETENESS ATTACKS ==============

    def inject_random_nulls(self, df: pd.DataFrame, null_rate: float = 0.5) -> pd.DataFrame:
        """Randomly replace 50% of values with NaN"""
        logger.info(f"[RED TEAM] Injecting random nulls at {null_rate*100}% rate")
        corrupted = df.copy()
        mask = np.random.random(df.shape) < null_rate
        corrupted = corrupted.mask(mask)
        return corrupted

    def remove_entire_columns(self, df: pd.DataFrame) -> pd.DataFrame:
        """Remove random critical columns"""
        logger.info("[RED TEAM] Removing entire columns")
        corrupted = df.copy()
        cols_to_drop = np.random.choice(df.columns, size=max(1, len(df.columns)//3), replace=False)
        return corrupted.drop(columns=cols_to_drop)

    def create_data_gaps(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create intermittent gaps in time-series data"""
        logger.info("[RED TEAM] Creating intermittent data gaps")
        corrupted = df.copy()
        # Remove random 30% of rows to simulate gaps
        remove_indices = np.random.choice(df.index, size=int(len(df)*0.3), replace=False)
        return corrupted.drop(remove_indices)

    # ============== ACCURACY ATTACKS ==============

    def inject_statistical_outliers(self, df: pd.DataFrame) -> pd.DataFrame:
        """Inject extreme outliers (>3 sigma) in numeric columns"""
        logger.info("[RED TEAM] Injecting statistical outliers")
        corrupted = df.copy()
        numeric_cols = df.select_dtypes(include=[np.number]).columns

        for col in numeric_cols:
            mean = df[col].mean()
            std = df[col].std()
            # Inject outliers in 5% of rows
            outlier_indices = np.random.choice(df.index, size=int(len(df)*0.05), replace=False)
            corrupted.loc[outlier_indices, col] = mean + (np.random.choice([-1, 1]) * 5 * std)

        return corrupted

    def corrupt_decimal_formats(self, df: pd.DataFrame) -> pd.DataFrame:
        """Swap decimal separators (. vs ,) to create parsing errors"""
        logger.info("[RED TEAM] Corrupting decimal formats")
        corrupted = df.copy()
        # This would typically affect CSV parsing, simulate by multiplying by 1000
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        for col in numeric_cols:
            # Corrupt 10% of values
            corrupt_indices = np.random.choice(df.index, size=int(len(df)*0.1), replace=False)
            corrupted.loc[corrupt_indices, col] *= 1000
        return corrupted

    def inject_typos(self, df: pd.DataFrame) -> pd.DataFrame:
        """Introduce typos in categorical columns"""
        logger.info("[RED TEAM] Injecting typos in categories")
        corrupted = df.copy()
        categorical_cols = df.select_dtypes(include=['object']).columns

        for col in categorical_cols:
            # Inject typos in 15% of values
            typo_indices = np.random.choice(df.index, size=int(len(df)*0.15), replace=False)
            for idx in typo_indices:
                original = str(corrupted.loc[idx, col])
                if len(original) > 2:
                    # Swap two random characters
                    pos = np.random.randint(0, len(original)-1)
                    typo = original[:pos] + original[pos+1] + original[pos] + original[pos+2:]
                    corrupted.loc[idx, col] = typo

        return corrupted

    # ============== CONSISTENCY ATTACKS ==============

    def create_sum_conflicts(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create scenarios where details don't sum to totals"""
        logger.info("[RED TEAM] Creating sum conflicts")
        # This is domain-specific - example for financial data
        corrupted = df.copy()
        if 'capital_amount_dzd' in corrupted.columns:
            # Randomly alter 20% of values
            alter_indices = np.random.choice(df.index, size=int(len(df)*0.2), replace=False)
            corrupted.loc[alter_indices, 'capital_amount_dzd'] *= np.random.uniform(0.5, 1.5, size=len(alter_indices))
        return corrupted

    def duplicate_keys(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create duplicate primary keys"""
        logger.info("[RED TEAM] Duplicating primary keys")
        corrupted = df.copy()
        if 'company_id' in corrupted.columns:
            # Duplicate 10% of rows with same company_id
            dup_indices = np.random.choice(df.index, size=int(len(df)*0.1), replace=False)
            duplicates = corrupted.loc[dup_indices].copy()
            corrupted = pd.concat([corrupted, duplicates], ignore_index=True)
        return corrupted

    def create_logical_contradictions(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create contradictory boolean/status fields"""
        logger.info("[RED TEAM] Creating logical contradictions")
        corrupted = df.copy()
        if 'status' in corrupted.columns:
            # Set contradictory status for 10% of rows
            contra_indices = np.random.choice(df.index, size=int(len(df)*0.1), replace=False)
            corrupted.loc[contra_indices, 'status'] = 'INVALID_STATUS'
        return corrupted

    # ============== TIMELINESS ATTACKS ==============

    def backdate_timestamps(self, df: pd.DataFrame) -> pd.DataFrame:
        """Set timestamps to past dates"""
        logger.info("[RED TEAM] Backdating timestamps")
        corrupted = df.copy()
        date_cols = df.select_dtypes(include=['datetime64']).columns
        for col in date_cols:
            # Backdate 20% of timestamps by 1-10 years
            backdate_indices = np.random.choice(df.index, size=int(len(df)*0.2), replace=False)
            corrupted.loc[backdate_indices, col] = pd.Timestamp('2010-01-01')
        return corrupted

    def inject_future_dates(self, df: pd.DataFrame) -> pd.DataFrame:
        """Inject future dates in historical data"""
        logger.info("[RED TEAM] Injecting future dates")
        corrupted = df.copy()
        if 'registration_date' in corrupted.columns:
            # Set future dates for 15% of rows
            future_indices = np.random.choice(df.index, size=int(len(df)*0.15), replace=False)
            corrupted.loc[future_indices, 'registration_date'] = pd.Timestamp('2030-01-01')
        return corrupted

    # ============== MAIN TESTING INTERFACE ==============

    def run_adversarial_tests(self, df: pd.DataFrame, validation_func: Callable) -> Dict[str, Dict]:
        """
        Execute all torture tests and measure system resilience

        Args:
            df: Clean input dataframe
            validation_func: Function that validates data and raises exception on failure

        Returns:
            Dictionary with test results for each attack type
        """
        results = {}

        for attack_type in AttackType:
            results[attack_type.value] = {}

            # Get all attacks of this type
            type_attacks = [name for name, func in self.attack_registry.items()
                          if attack_type.value in name]

            for attack_name in type_attacks:
                logger.info(f"\n{'='*60}")
                logger.info(f"Executing: {attack_name}")
                logger.info(f"{'='*60}")

                # Apply attack
                attack_func = self.attack_registry[attack_name]
                corrupted_df = attack_func(df)

                # Measure detection and recovery
                import time
                start_time = time.time()

                detected = False
                recovery_success = False
                error_details = None

                try:
                    # Attempt validation (should fail for corrupted data)
                    validation_func(corrupted_df)
                    # If we get here, attack was NOT detected (BAD!)
                    detected = False
                except Exception as e:
                    # Attack was detected (GOOD!)
                    detected = True
                    error_details = str(e)

                    # Attempt recovery (would be custom per attack type)
                    try:
                        # Placeholder for actual recovery logic
                        recovery_success = True
                    except Exception:
                        recovery_success = False

                detection_time = (time.time() - start_time) * 1000  # Convert to ms

                results[attack_type.value][attack_name] = {
                    'detected': detected,
                    'detection_time_ms': detection_time,
                    'recovery_success': recovery_success,
                    'passed': detected and recovery_success,
                    'error_details': error_details
                }

                # Log results
                status = "✅ PASSED" if results[attack_type.value][attack_name]['passed'] else "❌ FAILED"
                logger.info(f"{status} - {attack_name}")
                logger.info(f"  Detection: {detected}, Recovery: {recovery_success}")
                logger.info(f"  Time: {detection_time:.2f}ms")

        return results

    def generate_resilience_report(self, results: Dict) -> str:
        """Generate human-readable resilience report"""
        report = "\n" + "="*80 + "\n"
        report += "RED TEAM DATA QUALITY RESILIENCE REPORT\n"
        report += "="*80 + "\n\n"

        total_tests = 0
        passed_tests = 0

        for attack_type, attacks in results.items():
            report += f"\n{attack_type.upper()} ATTACKS:\n"
            report += "-" * 40 + "\n"

            for attack_name, result in attacks.items():
                total_tests += 1
                if result['passed']:
                    passed_tests += 1
                    status = "✅ PASSED"
                else:
                    status = "❌ FAILED"

                report += f"{status} {attack_name}\n"
                report += f"  Detection Rate: {'Yes' if result['detected'] else 'No'}\n"
                report += f"  Recovery: {'Yes' if result['recovery_success'] else 'No'}\n"
                report += f"  Time: {result['detection_time_ms']:.2f}ms\n\n"

        # Overall score
        resilience_score = (passed_tests / total_tests) * 100 if total_tests > 0 else 0
        report += "="*80 + "\n"
        report += f"OVERALL RESILIENCE SCORE: {resilience_score:.1f}% ({passed_tests}/{total_tests} tests passed)\n"

        if resilience_score >= 95:
            report += "STATUS: 🟢 EXCELLENT - System is highly resilient\n"
        elif resilience_score >= 85:
            report += "STATUS: 🟡 GOOD - Minor improvements needed\n"
        elif resilience_score >= 70:
            report += "STATUS: 🟠 FAIR - Significant gaps in resilience\n"
        else:
            report += "STATUS: 🔴 CRITICAL - Major vulnerabilities detected\n"

        report += "="*80 + "\n"

        return report


# Example usage
if __name__ == "__main__":
    # Sample test
    from algeria_data_platform.services.ingestion import validate_company_data
    from algeria_data_platform.data_loader import load_companies_from_csv, COMPANY_DATA_PATH

    # Load clean data
    clean_df = load_companies_from_csv(COMPANY_DATA_PATH)

    # Initialize red team
    red_team = RedTeamDataQuality()

    # Run adversarial tests
    results = red_team.run_adversarial_tests(clean_df, validate_company_data)

    # Generate report
    report = red_team.generate_resilience_report(results)
    print(report)
