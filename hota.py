import os
import numpy as np
from scipy.optimize import linear_sum_assignment
from ._base_metric import _BaseMetric
from .. import _timing


class HOTA(_BaseMetric):
    """Класс, имплементирующий HOTA метрику
    """
    def __init__(self):
        super().__init__()
        self.plottable = True
        self.array_labels = np.arange(0.05, 0.99, 0.05)
        self.integer_array_fields = ['HOTA_TP', 'HOTA_FN', 'HOTA_FP']
        self.float_array_fields = ['HOTA', 'DetA', 'AssA', 'DetRe', 'DetPr', 'AssRe', 'AssPr', 'LocA', 'RHOTA']
        self.float_fields = ['HOTA(0)', 'LocA(0)', 'HOTALocA(0)']
        self.fields = self.float_array_fields + self.integer_array_fields + self.float_fields
        self.summary_fields = self.float_array_fields + self.float_fields

    @_timing.time
    def eval_sequence(self, data):
        """Вычисляем HOTA метрику для последовательности"""

        # Инициализируем результат
        res = {}
        for field in self.float_array_fields + self.integer_array_fields:
            res[field] = np.zeros((len(self.array_labels)), dtype=np.float)
        for field in self.float_fields:
            res[field] = 0

        # Быстро возвращаем результат, если трекер или последовательность пусты
        if data['num_tracker_dets'] == 0:
            res['HOTA_FN'] = data['num_gt_dets'] * np.ones((len(self.array_labels)), dtype=np.float)
            res['LocA'] = np.ones((len(self.array_labels)), dtype=np.float)
            res['LocA(0)'] = 1.0
            return res
        if data['num_gt_dets'] == 0:
            res['HOTA_FP'] = data['num_tracker_dets'] * np.ones((len(self.array_labels)), dtype=np.float)
            res['LocA'] = np.ones((len(self.array_labels)), dtype=np.float)
            res['LocA(0)'] = 1.0
            return res

        # Подсчет переменных по глобальным ассоциациям
        potential_matches_count = np.zeros((data['num_gt_ids'], data['num_tracker_ids']))
        gt_id_count = np.zeros((data['num_gt_ids'], 1))
        tracker_id_count = np.zeros((1, data['num_tracker_ids']))

        # Сначала идем по каждому временному интервалу и собираем глобальную ифнормацию о треке
        for t, (gt_ids_t, tracker_ids_t) in enumerate(zip(data['gt_ids'], data['tracker_ids'])):
            # Считаем потенциальные совпадения между каждыми айдишниками на каждом временном шаге
            # При этом они нормализованы и взвешены с учетом сходства
            similarity = data['similarity_scores'][t]
            sim_iou_denom = similarity.sum(0)[np.newaxis, :] + similarity.sum(1)[:, np.newaxis] - similarity
            sim_iou = np.zeros_like(similarity)
            sim_iou_mask = sim_iou_denom > 0
            sim_iou[sim_iou_mask] = similarity[sim_iou_mask] / sim_iou_denom[sim_iou_mask]
            potential_matches_count[gt_ids_t[:, np.newaxis], tracker_ids_t[np.newaxis, :]] += sim_iou

            # Считаем общее количество "смертей" для gt_id и tracker_id
            gt_id_count[gt_ids_t] += 1
            tracker_id_count[0, tracker_ids_t] += 1

        # Считаем общую оценку соответствия индекса Жаккара между айдишниками
        global_alignment_score = potential_matches_count / (gt_id_count + tracker_id_count - potential_matches_count)
        matches_counts = [np.zeros_like(potential_matches_count) for _ in self.array_labels]

        # Считаем значения для каждого временного шага
        for t, (gt_ids_t, tracker_ids_t) in enumerate(zip(data['gt_ids'], data['tracker_ids'])):
            # Случай, когда на временном шаге нет gt_debt или tracker_debt
            if len(gt_ids_t) == 0:
                for a, alpha in enumerate(self.array_labels):
                    res['HOTA_FP'][a] += len(tracker_ids_t)
                continue
            if len(tracker_ids_t) == 0:
                for a, alpha in enumerate(self.array_labels):
                    res['HOTA_FN'][a] += len(gt_ids_t)
                continue

            # Получаем совпадающие значения между парами обнаружений для оптимизации HOTA
            similarity = data['similarity_scores'][t]
            score_mat = global_alignment_score[gt_ids_t[:, np.newaxis], tracker_ids_t[np.newaxis, :]] * similarity

            # Венгерский алгоритм
            match_rows, match_cols = linear_sum_assignment(-score_mat)

            # Вычисляем и сохраняем основные статистические данные
            for a, alpha in enumerate(self.array_labels):
                actually_matched_mask = similarity[match_rows, match_cols] >= alpha
                alpha_match_rows = match_rows[actually_matched_mask]
                alpha_match_cols = match_cols[actually_matched_mask]
                num_matches = len(alpha_match_rows)
                res['HOTA_TP'][a] += num_matches
                res['HOTA_FN'][a] += len(gt_ids_t) - num_matches
                res['HOTA_FP'][a] += len(tracker_ids_t) - num_matches
                if num_matches > 0:
                    res['LocA'][a] += sum(similarity[alpha_match_rows, alpha_match_cols])
                    matches_counts[a][gt_ids_t[alpha_match_rows], tracker_ids_t[alpha_match_cols]] += 1

        # Считаем значения AssA, AssRe, AssPr для alpha-значения
        # Считаем баллы за комбинацию gt_id и tracker_id, а после усредняем их по количеству обнаружений
        for a, alpha in enumerate(self.array_labels):
            matches_count = matches_counts[a]
            ass_a = matches_count / np.maximum(1, gt_id_count + tracker_id_count - matches_count)
            res['AssA'][a] = np.sum(matches_count * ass_a) / np.maximum(1, res['HOTA_TP'][a])
            ass_re = matches_count / np.maximum(1, gt_id_count)
            res['AssRe'][a] = np.sum(matches_count * ass_re) / np.maximum(1, res['HOTA_TP'][a])
            ass_pr = matches_count / np.maximum(1, tracker_id_count)
            res['AssPr'][a] = np.sum(matches_count * ass_pr) / np.maximum(1, res['HOTA_TP'][a])

        # Считаем финальное значение
        res['LocA'] = np.maximum(1e-10, res['LocA']) / np.maximum(1e-10, res['HOTA_TP'])
        res = self._compute_final_fields(res)
        return res

    def combine_sequences(self, all_res):
        """Комбинируем метрики для всех последовательностей"""
        res = {}
        for field in self.integer_array_fields:
            res[field] = self._combine_sum(all_res, field)
        for field in ['AssRe', 'AssPr', 'AssA']:
            res[field] = self._combine_weighted_av(all_res, field, res, weight_field='HOTA_TP')
        loca_weighted_sum = sum([all_res[k]['LocA'] * all_res[k]['HOTA_TP'] for k in all_res.keys()])
        res['LocA'] = np.maximum(1e-10, loca_weighted_sum) / np.maximum(1e-10, res['HOTA_TP'])
        res = self._compute_final_fields(res)
        return res

    @staticmethod
    def _compute_final_fields(res):
        """
        Считаем дополнительные метрики, которые зависят только от других значений подметрик.
        Функция используется как для подсчета каждой последовательности, так и для объединения значений в разных
        последовательностях
        """
        res['DetRe'] = res['HOTA_TP'] / np.maximum(1, res['HOTA_TP'] + res['HOTA_FN'])
        res['DetPr'] = res['HOTA_TP'] / np.maximum(1, res['HOTA_TP'] + res['HOTA_FP'])
        res['DetA'] = res['HOTA_TP'] / np.maximum(1, res['HOTA_TP'] + res['HOTA_FN'] + res['HOTA_FP'])
        res['HOTA'] = np.sqrt(res['DetA'] * res['AssA'])
        res['RHOTA'] = np.sqrt(res['DetRe'] * res['AssA'])

        res['HOTA(0)'] = res['HOTA'][0]
        res['LocA(0)'] = res['LocA'][0]
        res['HOTALocA(0)'] = res['HOTA(0)']*res['LocA(0)']
        return res

    def plot_single_tracker_results(self, table_res, tracker, cls, output_folder):
        """Создаем график для визуализации результатов"""
        from matplotlib import pyplot as plt
        res = table_res['COMBINED_SEQ']
        styles_to_plot = ['r', 'b', 'g', 'b--', 'b:', 'g--', 'g:', 'm']
        for name, style in zip(self.float_array_fields, styles_to_plot):
            plt.plot(self.array_labels, res[name], style)
        plt.xlabel('alpha')
        plt.ylabel('score')
        plt.title(tracker + ' - ' + cls)
        plt.axis([0, 1, 0, 1])
        legend = []
        for name in self.float_array_fields:
            legend += [name + ' (' + str(np.round(np.mean(res[name]), 2)) + ')']
        plt.legend(legend, loc='lower left')
        out_file = os.path.join(output_folder, cls + '_plot.pdf')
        os.makedirs(os.path.dirname(out_file), exist_ok=True)
        plt.savefig(out_file)
        plt.clf()