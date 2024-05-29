if __name__ == "__main__":
    exp = __import__('1_reconstruct_shape')
    exp.main()

    exp = __import__('2_align_augmented_shapes')
    exp.main()

    exp = __import__('3_synthesise_shape')
    exp.main()

    exp = __import__('4_construct_density_representation')
    exp.main()

    exp = __import__('6_register_template_to_shape')
    exp.main()

    exp = __import__('7_plot_shape_space')
    exp.main()

    exp = __import__('8_interpolate_between_shapes')
    exp.main()

    # exp = __import__('9_construct_atlas')
    # exp.main()

    exp = __import__('A_construct_sm')
    exp.main()

    exp = __import__('A_construct_sm_px')
    exp.main()