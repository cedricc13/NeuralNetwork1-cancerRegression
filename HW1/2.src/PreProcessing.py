import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import RobustScaler
import scipy.stats as stats


class CancerDataPreprocessor:
    """
    A comprehensive preprocessing class for the cancer dataset.
    Use this step by step to clean and prepare your data for machine learning.
    """
    def __init__(self, data_file='cancer_reg-1.csv'):
        """Initialize the preprocessor with the dataset file"""
        self.data_file = data_file
        self.df = pd.read_csv(self.data_file, encoding='latin1')
        self.X = None
        self.y = None
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        self.scaler = None
        self.target_column = 'TARGET_deathRate'
        self.log_transformed_features = []  # Track which features have been log-transformed

    #Special function to answer a question of STEP 1
    def show_min_max_table(self):
        print("\n📊 TABLEAU MIN/MAX DES FEATURES")
        print("=" * 60)

        if self.df is None:
            self.df = pd.read_csv(self.data_file, encoding='latin1')
        
        numeric_cols = self.df.select_dtypes(include=[np.number]).columns
        
        if self.target_column in numeric_cols:
            numeric_cols = numeric_cols.drop(self.target_column)
        
        min_max_data = []
        for col in numeric_cols:
            min_val = self.df[col].min()
            max_val = self.df[col].max()
            
            min_max_data.append({
                'Feature': col,
                'Min': f"{min_val:.2f}" if pd.notnull(min_val) else "NaN",
                'Max': f"{max_val:.2f}" if pd.notnull(max_val) else "NaN", 
            })
        
        min_max_df = pd.DataFrame(min_max_data)
        
        print(f"{'Feature':<25} {'Min':<12} {'Max':<12}")
        print("-" * 85)
        
        for _, row in min_max_df.iterrows():
            print(f"{row['Feature']:<25} {row['Min']:<12} {row['Max']:<12}")
        
        print(f"\n✅ Total features: {len(numeric_cols)}")
        
        return min_max_df
    
    def step1_missing_values(self, strategy='median'):
        """Step 1: Handle missing values in numeric columns only"""
        print(f"\nSTEP 1: Missing Values Analysis")
        print("=" * 60)
        
        # Drop column with too many missing values
        if "PctSomeCol18_24" in self.df.columns:
            self.df.drop("PctSomeCol18_24", axis=1, inplace=True)
        
        # Get numeric columns only
        numeric_cols = self.df.select_dtypes(include=[np.number]).columns
        
        # Apply strategy only to numeric columns
        if strategy == 'mean':
            self.df[numeric_cols] = self.df[numeric_cols].fillna(self.df[numeric_cols].mean())
        elif strategy == 'median':
            self.df[numeric_cols] = self.df[numeric_cols].fillna(self.df[numeric_cols].median())
        elif strategy == 'mode':
            for col in numeric_cols:
                self.df[col] = self.df[col].fillna(self.df[col].mode()[0] if not self.df[col].mode().empty else 0)
        
        missing_after = self.df[numeric_cols].isnull().sum().sum()
        print(f"Missing values in numeric columns after treatment: {missing_after}")
        
        return

    def step2CategoryVariablesHandling(self, keep_Geography=False, keep_binnedInc=False):
        """Step 2: Handle categorical variables"""
        print(f"\n STEP 2: Categorical Variables Handling")
        print("=" * 60)
        if not keep_Geography:
            self.df.drop("Geography", axis=1, inplace=True) #drop the column Geography (location and no value)
            print("Column Geography dropped")
        else:
            return
        if not keep_binnedInc:
            self.df.drop("binnedInc", axis=1, inplace=True) #drop the column binnedInc (more than 70% of missing values)
            print("Column binnedInc dropped")
        else:
            le = LabelEncoder()
            self.df["binnedInc"] = le.fit_transform(self.df["binnedInc"])
        return

    def step3_remove_outliers(self):
        """Step 3: Remove outliers"""
        zscore = self.df.select_dtypes(include=[np.number]).columns
        for col in zscore:
            z_scores = np.abs(stats.zscore(self.df[col]))
            col_outliers = z_scores > 3
            self.df.loc[col_outliers, col] = self.df[col].mean()
            print(f"Total values replaced: {col_outliers.sum()}")
        return

    def step4_log_transform(self, 
                           method='log1p', 
                           auto_select=True, 
                           skewness_threshold=2.0,
                           specific_features=None,
                           handle_negatives='shift'):
        """
        Step 4: Apply log transformation to reduce skewness
        
        Parameters:
        -----------
        method : str
            Type of log transform ('log', 'log10', 'log1p', 'log2')
        auto_select : bool
            Automatically select features based on skewness
        skewness_threshold : float
            Threshold for automatic feature selection (default: 2.0)
        specific_features : list
            Specific features to transform (overrides auto_select)
        handle_negatives : str
            How to handle negative values ('shift', 'skip', 'abs')
        """
        print(f"\nSTEP 4: Log Transformation")
        print("=" * 50)
        print(f"Method: {method}")
        print(f"Auto-select features: {auto_select}")
        print(f"Skewness threshold: {skewness_threshold}")
        print(f"Handle negatives: {handle_negatives}")
        
        # Get numeric columns (excluding target)
        numeric_cols = self.df.select_dtypes(include=[np.number]).columns
        feature_cols = [col for col in numeric_cols if col != self.target_column]
        
        if len(feature_cols) == 0:
            print("No numeric features found for transformation")
            return
        
        # Determine which features to transform
        features_to_transform = []
        
        if specific_features is not None:
            # Use specific features provided
            features_to_transform = [f for f in specific_features if f in feature_cols]
            print(f"Using specific features: {features_to_transform}")
        elif auto_select:
            # Auto-select based on skewness
            print(f"\nAnalyzing skewness (threshold: {skewness_threshold}):")
            print("-" * 50)
            
            for col in feature_cols:
                if self.df[col].min() >= 0 or handle_negatives != 'skip':  # Can handle this column
                    skew = abs(self.df[col].skew())
                    if skew > skewness_threshold:
                        features_to_transform.append(col)
                        print(f"  ✅ {col:<25}: skewness = {skew:.3f} (WILL TRANSFORM)")
                    else:
                        print(f"  ⏭️  {col:<25}: skewness = {skew:.3f} (skip)")
                else:
                    print(f"  ❌ {col:<25}: has negative values (skip)")
        else:
            # Transform all suitable features
            for col in feature_cols:
                if self.df[col].min() >= 0 or handle_negatives != 'skip':
                    features_to_transform.append(col)
        
        if not features_to_transform:
            print("\n⚠️  No features selected for log transformation")
            return
        
        print(f"\n🔄 Applying {method} transformation to {len(features_to_transform)} features:")
        print("-" * 60)
        
        transformed_count = 0
        for col in features_to_transform:
            try:
                original_data = self.df[col].copy()
                
                # Handle negative values
                if self.df[col].min() < 0:
                    if handle_negatives == 'shift':
                        # Shift data to make all values positive
                        shift_amount = abs(self.df[col].min()) + 1
                        self.df[col] = self.df[col] + shift_amount
                        print(f"  📈 {col}: shifted by +{shift_amount:.2f} to handle negatives")
                    elif handle_negatives == 'abs':
                        # Take absolute values
                        self.df[col] = self.df[col].abs()
                        print(f"  📈 {col}: took absolute values")
                    elif handle_negatives == 'skip':
                        print(f"  ⏭️  {col}: skipped (negative values)")
                        continue
                
                # Handle zeros for log (not log1p)
                if method in ['log', 'log10', 'log2'] and (self.df[col] == 0).any():
                    # Add small constant to avoid log(0)
                    epsilon = 1e-8
                    self.df[col] = self.df[col] + epsilon
                    print(f"  📈 {col}: added epsilon ({epsilon}) to handle zeros")
                
                # Apply transformation
                if method == 'log' or method == 'ln':
                    self.df[col] = np.log(self.df[col])
                elif method == 'log10':
                    self.df[col] = np.log10(self.df[col])
                elif method == 'log1p':
                    self.df[col] = np.log1p(self.df[col])  # log(1 + x)
                elif method == 'log2':
                    self.df[col] = np.log2(self.df[col])
                else:
                    raise ValueError(f"Unknown log method: {method}")
                
                # Calculate improvement in skewness
                original_skew = abs(original_data.skew())
                new_skew = abs(self.df[col].skew())
                improvement = original_skew - new_skew
                
                print(f"  ✅ {col:<25}: skew {original_skew:.3f} → {new_skew:.3f} "
                      f"(Δ: {improvement:+.3f})")
                
                transformed_count += 1
                self.log_transformed_features.append(col)
                
            except Exception as e:
                print(f"  ❌ {col}: transformation failed - {str(e)}")
                # Restore original data if transformation failed
                self.df[col] = original_data
        
        print(f"\n📊 LOG TRANSFORMATION SUMMARY:")
        print(f"  Features transformed: {transformed_count}/{len(features_to_transform)}")
        print(f"  Method used: {method}")
        print(f"  Transformed features: {self.log_transformed_features}")
        
        return

    def step5_split_data(self, test_size=0.15, val_size=0.15, random_state=42, shuffle=False):
        """Step 5: Split data into train, validation, and test sets"""
        print(f"\nSTEP 5: Splitting Data")
        print("=" * 50)
        print(f"Shuffle data: {shuffle}")
        
        if val_size > 0:
            # Three-way split: train, val, test
            X_temp, self.X_test, y_temp, self.y_test = train_test_split(
                self.X, self.y, test_size=test_size, random_state=random_state, shuffle=shuffle
            )
            
            val_size_adjusted = val_size / (1 - test_size)
            self.X_train, self.X_val, self.y_train, self.y_val = train_test_split(
                X_temp, y_temp, test_size=val_size_adjusted, random_state=random_state, shuffle=shuffle
            )
            
            print(f"Training set: {self.X_train.shape}")
            print(f"Validation set: {self.X_val.shape}")
            print(f"Test set: {self.X_test.shape}")
            
            return self.X_train, self.X_val, self.X_test, self.y_train, self.y_val, self.y_test
        else:
            # Two-way split: train, test
            self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
                self.X, self.y, test_size=test_size, random_state=random_state, shuffle=shuffle
            )
            
            print(f"Training set: {self.X_train.shape}")
            print(f"Test set: {self.X_test.shape}")
            
            return self.X_train, self.X_test, self.y_train, self.y_test
            
    def step6_scale_features(self, method='robust'):
        """Step 6: Scale/normalize features"""
        print(f"\n📏 STEP 6: Feature Scaling (Method: {method})")
        print("=" * 50)
        
        if method == 'standard':
            self.scaler = StandardScaler()
        elif method == 'minmax':
            self.scaler = MinMaxScaler()
        elif method == 'robust':
            self.scaler = RobustScaler()

        else:
            raise ValueError("Method must be 'standard' or 'minmax' or 'robust'")
        
        # Fit on training data only
        self.X_train_scaled = self.scaler.fit_transform(self.X_train)
        self.X_test_scaled = self.scaler.transform(self.X_test)
        
        if hasattr(self, 'X_val'):
            self.X_val_scaled = self.scaler.transform(self.X_val)
            print("✅ Training, validation, and test sets scaled!")
            return self.X_train_scaled, self.X_val_scaled, self.X_test_scaled
        else:
            print("✅ Training and test sets scaled!")
            return self.X_train_scaled, self.X_test_scaled


    def step7_feature_summary(self,):
        """Step 7: Final summary of preprocessed features"""
        print("\n📋 STEP 7: Preprocessing Summary")
        print("=" * 50)
        
        print(f"✅ Original dataset: {self.df.shape}")
        print(f"✅ Final features: {self.X.shape}")
        print(f"✅ Training samples: {self.X_train.shape[0]}")
        print(f"✅ Test samples: {self.X_test.shape[0]}")
        
        if hasattr(self, 'X_val'):
            print(f"✅ Validation samples: {self.X_val.shape[0]}")
        
        print(f"✅ Feature names ({len(self.X.columns)}):")
        for i, col in enumerate(self.X.columns, 1):
            print(f"  {i:2}. {col}")
        
        print("\n🎉 Data preprocessing completed successfully!")
        
        return {
            'feature_names': list(self.X.columns),
            'n_features': self.X.shape[1],
            'n_samples': self.X.shape[0],
            'train_size': self.X_train.shape[0],
            'test_size': self.X_test.shape[0]
        }
    
    def run_all_steps(self, 
                      missing_strategy='median',
                      keep_geography=False,
                      keep_binned_inc=True,
                      outlier_strategy='replace_mean',
                      scaling_method='robust',
                      test_size=0.15,
                      val_size=0.15,
                      shuffle=True,
                      random_state=42):
        
        self.step1_missing_values(strategy=missing_strategy)
        self.step2CategoryVariablesHandling(keep_Geography=keep_geography, keep_binnedInc=keep_binned_inc)
        self.step3_remove_outliers()
        self.step4_log_transform()
        
        # Prepare X and y from processed dataframe
        self.y = self.df[self.target_column]
        self.X = self.df.drop(self.target_column, axis=1)
        
        print(f"\n📊 Features and target prepared:")
        print(f"   Features shape: {self.X.shape}")
        print(f"   Target shape: {self.y.shape}")

        split_results = self.step5_split_data(test_size=test_size, val_size=val_size, shuffle=shuffle, random_state=random_state)
        self.step6_scale_features(method=scaling_method)
        final_summary = self.step7_feature_summary()
        
        # Return summary
        return {
            'preprocessing_completed': True,
            'original_shape': self.df.shape,
            'final_shape': self.X.shape,
            'train_samples': self.X_train.shape[0],
            'val_samples': self.X_val.shape[0],
            'test_samples': self.X_test.shape[0],
            'features': self.X.shape[1],
            'split_results': split_results,
            'final_summary': final_summary
        }


if __name__ == "__main__":
    # Créer l'instance
    preprocessor = CancerDataPreprocessor('cancer_reg-1.csv')
    preprocessor.run_all_steps()
    preprocessor.step7_feature_summary()