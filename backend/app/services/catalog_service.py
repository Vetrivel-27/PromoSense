"""Catalog and Elasticity Service for PromoSense Backend."""

from backend.app.schemas.common import MethodologicalWarnings
from backend.app.schemas.products import (
    ProductCatalogResponse,
    ProductDetailResponse,
    ProductElasticityResponse,
    ProductSummary,
    SubstituteElasticityDetail,
    SubstitutePairSummary,
)
from backend.app.services.model_registry import ModelRegistry


class CatalogService:
    """Service providing product catalog listings and elasticity diagnostics."""

    def __init__(self, registry: ModelRegistry | None = None) -> None:
        self.registry = registry or ModelRegistry.get_instance()

    def get_catalog(self) -> ProductCatalogResponse:
        """Retrieves list of target products and substitute pairs with empirical baselines."""
        if not self.registry.is_initialized:
            self.registry.load_artifacts()

        product_summaries: list[ProductSummary] = []
        val_df = self.registry.val_df

        # Build pair lookups
        pairs = self.registry.substitute_pairs
        focal_a_map: dict[int, list[str]] = {}
        substitute_b_map: dict[int, list[str]] = {}

        for p in pairs:
            p_name = p["pair_name"]
            focal_a_map.setdefault(p["prod_A"], []).append(p_name)
            substitute_b_map.setdefault(p["prod_B"], []).append(p_name)

        for pid in self.registry.target_skus:
            meta = self.registry.get_product_metadata(pid)
            p_df = val_df[val_df["product_id"] == pid]

            avg_shelf = float(p_df["shelf_price"].mean()) if len(p_df) > 0 else 0.0
            avg_disp = float(p_df["display_share"].mean()) if len(p_df) > 0 else 0.0
            avg_mail = float(p_df["mailer_share"].mean()) if len(p_df) > 0 else 0.0

            associated_pairs = list(set(focal_a_map.get(pid, []) + substitute_b_map.get(pid, [])))

            product_summaries.append(
                ProductSummary(
                    product_id=pid,
                    commodity=meta["commodity"],
                    sub_commodity=meta["sub_commodity"],
                    brand_type=meta["brand_type"],
                    department=meta["department"],
                    baseline_shelf_price=round(avg_shelf, 4),
                    observed_display_share=round(avg_disp, 4),
                    observed_mailer_share=round(avg_mail, 4),
                    substitute_pair_names=associated_pairs,
                    is_focal_substitute_a=bool(pid in focal_a_map),
                    is_substitute_b=bool(pid in substitute_b_map),
                )
            )

        pair_summaries: list[SubstitutePairSummary] = []
        p6_cannibalization = self.registry.phase_6_results.get("cannibalization_analysis", {})

        for p in pairs:
            p_name = p["pair_name"]
            p_info = p6_cannibalization.get(p_name, {})
            is_weak = bool(
                "Milk" in p_name or p_info.get("is_weakly_identified_cross_price", False)
            )

            pair_summaries.append(
                SubstitutePairSummary(
                    pair_name=p_name,
                    commodity=p_info.get("commodity", "UNKNOWN"),
                    sub_commodity=p_info.get("sub_commodity", "UNKNOWN"),
                    prod_A_id=p["prod_A"],
                    prod_B_id=p["prod_B"],
                    is_weakly_identified=is_weak,
                )
            )

        return ProductCatalogResponse(
            products=product_summaries,
            substitute_pairs=pair_summaries,
            total_products=len(product_summaries),
        )

    def get_product_detail(self, product_id: int) -> ProductDetailResponse:
        """Retrieves detailed product metadata, historical baseline prices, volume, and substitute relations."""
        if not self.registry.is_initialized:
            self.registry.load_artifacts()

        if product_id not in self.registry.target_skus:
            raise KeyError(
                f"Product ID {product_id} not found in modeled catalog. Supported SKUs: {self.registry.target_skus}"
            )

        meta = self.registry.get_product_metadata(product_id)
        val_df = self.registry.val_df
        p_df = val_df[val_df["product_id"] == product_id]

        avg_shelf = float(p_df["shelf_price"].mean()) if len(p_df) > 0 else 0.0
        min_shelf = float(p_df["shelf_price"].min()) if len(p_df) > 0 else 0.0
        max_shelf = float(p_df["shelf_price"].max()) if len(p_df) > 0 else 0.0
        avg_disp = float(p_df["display_share"].mean()) if len(p_df) > 0 else 0.0
        avg_mail = float(p_df["mailer_share"].mean()) if len(p_df) > 0 else 0.0
        vol_mean = (
            float(p_df["quantity"].mean()) if "quantity" in p_df.columns and len(p_df) > 0 else 0.0
        )
        vol_std = (
            float(p_df["quantity"].std()) if "quantity" in p_df.columns and len(p_df) > 1 else 0.0
        )

        pairs = self.registry.substitute_pairs
        associated_pairs = []
        pair_summaries: list[SubstitutePairSummary] = []
        p6_cannibalization = self.registry.phase_6_results.get("cannibalization_analysis", {})

        is_focal_a = False
        is_sub_b = False

        for p in pairs:
            p_name = p["pair_name"]
            if p["prod_A"] == product_id:
                is_focal_a = True
                associated_pairs.append(p_name)
            if p["prod_B"] == product_id:
                is_sub_b = True
                associated_pairs.append(p_name)

            if p["prod_A"] == product_id or p["prod_B"] == product_id:
                p_info = p6_cannibalization.get(p_name, {})
                is_weak = bool(
                    "Milk" in p_name or p_info.get("is_weakly_identified_cross_price", False)
                )
                pair_summaries.append(
                    SubstitutePairSummary(
                        pair_name=p_name,
                        commodity=p_info.get("commodity", meta["commodity"]),
                        sub_commodity=p_info.get("sub_commodity", meta["sub_commodity"]),
                        prod_A_id=p["prod_A"],
                        prod_B_id=p["prod_B"],
                        is_weakly_identified=is_weak,
                    )
                )

        import math

        std_val = 0.0 if math.isnan(vol_std) else vol_std

        return ProductDetailResponse(
            product_id=product_id,
            commodity=meta["commodity"],
            sub_commodity=meta["sub_commodity"],
            brand_type=meta["brand_type"],
            department=meta["department"],
            baseline_shelf_price=round(avg_shelf, 4),
            observed_display_share=round(avg_disp, 4),
            observed_mailer_share=round(avg_mail, 4),
            substitute_pair_names=list(set(associated_pairs)),
            is_focal_substitute_a=is_focal_a,
            is_substitute_b=is_sub_b,
            min_observed_price=round(min_shelf, 4),
            max_observed_price=round(max_shelf, 4),
            historical_weekly_volume_mean=round(vol_mean, 2),
            historical_weekly_volume_std=round(std_val, 2),
            substitute_pair_details=pair_summaries,
        )

    def get_product_elasticity(self, product_id: int) -> ProductElasticityResponse:
        """Retrieves frozen elasticity estimates, confidence intervals, and model comparisons."""
        if not self.registry.is_initialized:
            self.registry.load_artifacts()

        p6_own = self.registry.phase_6_results.get("own_price_elasticities", {})
        str_pid = str(product_id)
        if str_pid not in p6_own:
            raise KeyError(
                f"Product ID {product_id} not found in frozen elasticity results. Supported SKUs: {self.registry.target_skus}"
            )

        p_data = p6_own[str_pid]
        models = p_data.get("models", {})

        # Champion model is controlled_paid_ols
        champ = models.get("controlled_paid_ols", {})
        champ_val = champ.get("val_eval", {})

        eps = float(champ.get("elasticity", 0.0))
        hac_se = float(champ_val.get("elasticity_hac_se", champ.get("elasticity_se", 0.0)))
        p_val = float(champ_val.get("p_val", 0.0))
        ci_low = round(eps - 1.96 * hac_se, 4)
        ci_high = round(eps + 1.96 * hac_se, 4)

        # Economic interpretation
        if abs(eps) < 0.2:
            interp = (
                f"Completely price inelastic (eps = {eps:.2f}). Unit sales do not respond to price changes; "
                "price discounts destroy contribution profit without expanding volume."
            )
        elif abs(eps) < 1.0:
            interp = (
                f"Inelastic demand (eps = {eps:.2f}). A 10% price cut increases sales by only {abs(eps) * 10:.1f}%, "
                "failing to compensate for price dilution and leading to negative revenue and profit lift."
            )
        else:
            interp = (
                f"Highly price-elastic demand (eps = {eps:.2f}). A 10% price cut drives strong volume expansion "
                f"of +{abs(eps) * 10:.1f}%. However, promotional profitability depends heavily on whether volume "
                "gain outpaces percentage unit margin dilution."
            )

        # Substitute pair cross-elasticity details
        sub_detail: SubstituteElasticityDetail | None = None
        cross_price_warning: str | None = None

        p6_cannibalization = self.registry.phase_6_results.get("cannibalization_analysis", {})
        for pair_name, p_info in p6_cannibalization.items():
            if p_info.get("prod_A") == product_id:
                elasticities = p_info.get("elasticities", {})
                eps_ab = float(elasticities.get("eps_AB_cross", 0.0))
                eps_ab_se = float(elasticities.get("eps_AB_se", 0.0))
                is_weak = bool("Milk" in pair_name or eps_ab_se > 0.5)

                if is_weak:
                    id_note = (
                        "Weakly identified cross-price elasticity (large standard error / collinear pricing). "
                        "Cross-price offsets represent model-implied illustrative scenario tracing, NOT verified causal cannibalization."
                    )
                    cross_price_warning = id_note
                else:
                    id_note = (
                        f"Well-identified substitute cross-elasticity (eps_AB = +{eps_ab:.3f}). "
                        f"A 10% price cut on Product A decreases Product B demand by approximately {abs(eps_ab) * 10:.1f}%."
                    )

                sub_detail = SubstituteElasticityDetail(
                    pair_name=pair_name,
                    substitute_product_id=p_info.get("prod_B"),
                    own_elasticity_A=float(elasticities.get("eps_AA_own", eps)),
                    cross_elasticity_AB=round(eps_ab, 4),
                    cross_elasticity_AB_se=round(eps_ab_se, 4),
                    is_weakly_identified=is_weak,
                    identification_note=id_note,
                )
                break

        methodological_warnings = MethodologicalWarnings(
            cross_price_warning=cross_price_warning,
        )

        meta = self.registry.get_product_metadata(product_id)

        return ProductElasticityResponse(
            product_id=product_id,
            commodity=meta["commodity"],
            sub_commodity=meta["sub_commodity"],
            brand_type=meta["brand_type"],
            champion_specification="PaidPriceElasticityOLS (Controlled Specification with Merchandising and Seasonality)",
            champion_elasticity=round(eps, 4),
            champion_hac_se=round(hac_se, 4),
            champion_ci_95=[ci_low, ci_high],
            champion_p_value=p_val,
            interpretation=interp,
            competing_models=models,
            substitute_pair_elasticity=sub_detail,
            methodological_warnings=methodological_warnings,
        )
