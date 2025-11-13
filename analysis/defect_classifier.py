class DefectClassifier:
    def classify(self, mr, outputs):
        # 比对MR期望与输出
        if not self._match(outputs, mr.expected_behavior):
            return "Fail"
        return "Pass"
