import pandas as pd
from scipy import stats

class HypothesisTesting:
    def __init__(self, data, verbose=True):
        self.data = pd.DataFrame(data)
        self.verbose = verbose
        self.alpha = 0.05

    def test_weekend_vs_weekday(self):
        if self.verbose:
            print("\n======================================================")
            print("   HYPOTHESIS TEST 1: Weekend vs. Weekday Delays")
            print("======================================================")

        if not all(col in self.data.columns for col in ["ARR_DELAY", "IS_WEEKEND"]):
            print("Error: Missing required columns 'ARR_DELAY' or 'IS_WEEKEND'.")
            return self

        weekday_delays = self.data[self.data['IS_WEEKEND'] == 0]['ARR_DELAY'].dropna()
        weekend_delays = self.data[self.data['IS_WEEKEND'] == 1]['ARR_DELAY'].dropna()

        stat, p_value = stats.ttest_ind(weekend_delays, weekday_delays, equal_var=False, alternative='two-sided')

        mean_weekday = weekday_delays.mean()
        mean_weekend = weekend_delays.mean()

        if self.verbose:
            print("Welch Two Sample t-test")
            print(f"t = {stat:.4f}, p-value = {p_value:.4g}")
            print("Sample estimates:")
            print(f"Mean of weekday (0): {mean_weekday:.4f}")
            print(f"Mean of weekend (1): {mean_weekend:.4f}")

        print("\n--- Conclusion ---")
        if p_value < self.alpha:
            print(f"Result: REJECT the null hypothesis (p = {p_value:.4g} < {self.alpha}).")
            if mean_weekend > mean_weekday:
                print("Conclusion: Weekends have significantly HIGHER arrival delays than weekdays.")
            else:
                print("Conclusion: Weekends have significantly LOWER arrival delays than weekdays.")
        else:
            print(f"Result: FAIL TO REJECT the null hypothesis (p = {p_value:.4g} >= {self.alpha}).")
            print("Conclusion: There is no significant difference in delays between weekends and weekdays.")

        return self

    def test_pandemic_impact(self):
        if self.verbose:
            print("\n======================================================")
            print("   HYPOTHESIS TEST 2: Pandemic Impact on Delays")
            print("======================================================")

        if not all(col in self.data.columns for col in ["ARR_DELAY", "FL_YEAR"]):
            print("Error: Missing required columns 'ARR_DELAY' or 'FL_YEAR'.")
            return self

        pre_delays = self.data[self.data['FL_YEAR'] == 2019]['ARR_DELAY'].dropna()
        post_delays = self.data[self.data['FL_YEAR'].isin([2022, 2023])]['ARR_DELAY'].dropna()

        if len(post_delays) == 0 or len(pre_delays) == 0:
            print("Error: Not enough data for the specified years to run the test.")
            return self

        stat, p_value = stats.ttest_ind(post_delays, pre_delays, equal_var=False, alternative='greater')

        mean_pre = pre_delays.mean()
        mean_post = post_delays.mean()

        if self.verbose:
            print("Welch Two Sample t-test (One-Sided: Post > Pre)")
            print(f"t = {stat:.4f}, p-value = {p_value:.4g}")
            print("Sample estimates:")
            print(f"Mean of Pre-Pandemic (2019): {mean_pre:.4f}")
            print(f"Mean of Post-Pandemic (2022-2023): {mean_post:.4f}")

        print("\n--- Conclusion ---")
        if p_value < self.alpha:
            print(f"Result: REJECT the null hypothesis (p = {p_value:.4g} < {self.alpha}).")
            print("Conclusion: Post-pandemic operations (2022-2023) suffered significantly more delays than pre-pandemic (2019) operations.")
        else:
            print(f"Result: FAIL TO REJECT the null hypothesis (p = {p_value:.4g} >= {self.alpha}).")
            print("Conclusion: Post-pandemic delays are NOT significantly higher than pre-pandemic delays.")

        return self

    def run_all_tests(self):
        self.test_weekend_vs_weekday().test_pandemic_impact()
        return self