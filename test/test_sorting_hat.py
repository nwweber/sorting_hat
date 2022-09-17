from pathlib import Path
from typing import List

import pandas
import pytest
from ortools.sat.python import cp_model
from ortools.sat.python.cp_model import BoundedLinearExpression, IntVar
from pandas import DataFrame

import sorting_hat
from sorting_hat import CourseAssignmentVariables, CourseRegistry, Student, StudentsRegistry


@pytest.fixture
def course_info():
    return DataFrame([{"name": "course_1", "min_size": 0, "max_size": 1}])


# unsure how to run test, breaks due to not finding example files at path given
# def test_has_example_problem():
#     student_course_preferences: StudentPreferences
#     course_max_nr_students: CourseCapacity
#     student_course_preferences, course_max_nr_students = sorting_hat.get_example_problem()
#     assert len(student_course_preferences) > 0
#     assert len(course_max_nr_students) > 0


def test_creates_assignment_variables():
    students: StudentsRegistry = StudentsRegistry([Student(preferences=["course_1"])])
    courses: CourseRegistry = CourseRegistry(
        DataFrame(
            [
                {"name": "course_1", "min_size": 0, "max_size": 1},
                {"name": "course_2", "min_size": 0, "max_size": 1},
            ]
        )
    )
    model = cp_model.CpModel()
    course_assignments: CourseAssignmentVariables = sorting_hat.generate_course_assignment_variables(
        students, courses, model
    )
    cp_vars: List[IntVar] = course_assignments.get_all()
    # creates one variable for each combination of student and course
    assert len(cp_vars) == len(students) * len(courses)


def test_gets_cp_variables_by_course_name():
    model: cp_model.CpModel = cp_model.CpModel()
    assignment_data = [
        (0, "course_1", model.NewIntVar(0, 1, "alice in course_1")),
        (0, "course_2", model.NewIntVar(0, 1, "alice in course_2")),
    ]
    variables: CourseAssignmentVariables = CourseAssignmentVariables(assignment_data)
    var_list_expected: List[IntVar] = [assignment_data[0][2]]
    var_list_returned: List[IntVar] = variables.get_by_course_name("course_1")
    assert var_list_returned == var_list_expected


def test_makes_cost_expression():
    students: StudentsRegistry = StudentsRegistry(
        [Student(preferences=["course_1", "course_2"])]
    )
    courses: CourseRegistry = CourseRegistry(
        DataFrame(
            [
                {"name": "course_1", "min_size": 0, "max_size": 1},
                {"name": "course_2", "min_size": 0, "max_size": 1},
                {"name": "course_3", "min_size": 0, "max_size": 1},
            ]
        )
    )
    model = cp_model.CpModel()
    course_assignments: CourseAssignmentVariables = sorting_hat.generate_course_assignment_variables(
        students, courses, model
    )
    alice_in_c1, alice_in_c2 = course_assignments.get_by_student_id_and_courses(
        0, ["course_1", "course_2"]
    )
    # note that 'course_3' does not even appear here. we make sure alice can never be assigned to a course she has not listed
    # as a preference through other constraints
    expected_cost: BoundedLinearExpression = 0 * alice_in_c1 + 1 * alice_in_c2
    cost: BoundedLinearExpression = sorting_hat.generate_cost(
        students, course_assignments
    )
    assert cost == expected_cost


def test_gets_all_assignments():
    model: cp_model.CpModel = cp_model.CpModel()
    assignment_data = [
        (0, "course_1", model.NewIntVar(0, 1, "alice in course_1")),
        (0, "course_2", model.NewIntVar(0, 1, "alice in course_2")),
    ]
    assignments: CourseAssignmentVariables = CourseAssignmentVariables(assignment_data)
    all_assignments_expected: List[IntVar] = [data[2] for data in assignment_data]
    all_assignments: List[IntVar] = assignments.get_all()
    assert all_assignments == all_assignments_expected


def test_reads_student_preference_file():
    pref_file_path: Path = Path("student_preferences.csv")
    student_registry: StudentsRegistry = sorting_hat.read_student_preferences_file(
        pref_file_path, None
    )
    assert len(student_registry) > 0


def test_reads_course_capacity_file():
    capacity_file_path: Path = Path("course_capacity.csv")
    capacities: CourseRegistry = sorting_hat.CourseRegistry.make_from_file(capacity_file_path, None)
    expected: CourseRegistry = CourseRegistry(
        DataFrame(
            [
                {"name": "course_1", "min_size": 0, "max_size": 1},
                {"name": "course_2", "min_size": 0, "max_size": 1},
                {"name": "course_3", "min_size": 0, "max_size": 1},
                {
                    "name": "Difficult, Course, With, Commas, Name",
                    "min_size": 0,
                    "max_size": 10,
                },
            ]
        )
    )
    assert capacities == expected


def all_courses_respect_min_nr_students(solution: DataFrame, courses: CourseRegistry) -> bool:
    student_counts_by_course: pandas.Series = solution.value_counts("course")
    course: str
    for course, student_count in student_counts_by_course.iteritems():
        min_student_count: int = courses.get_min_students_by_course_name(course)
        if student_count < min_student_count:
            return False
    return True


def all_courses_respect_max_nr_students(solution: DataFrame, courses: CourseRegistry):
    student_counts_by_course: pandas.Series = solution.value_counts("course")
    course: str
    for course, student_count in student_counts_by_course.iteritems():
        max_student_count: int = courses.get_max_students_by_course_name(course)
        if student_count > max_student_count:
            return False
    return True


def test_all_courses_respect_min_nr_students():
    courses = CourseRegistry(
        DataFrame(
            [
                {"name": "c1", "min_size": 2, "max_size": 3},
                {"name": "no_one_assigned_to_this", "min_size": 2, "max_size": 3},
            ]
        )
    )
    solution_fails: DataFrame = DataFrame(
        [{"student": "alice", "course": "c1"},]
    )
    solution_passes: DataFrame = DataFrame(
        [{"student": "alice", "course": "c1"}, {"student": "bob", "course": "c1"},]
    )
    assert not all_courses_respect_min_nr_students(solution_fails, courses)
    assert all_courses_respect_min_nr_students(solution_passes, courses)


def test_all_courses_respect_max_nr_students():
    courses = CourseRegistry(
        DataFrame(
            [
                {"name": "c1", "min_size": 2, "max_size": 3},
                {"name": "no_one_assigned_to_this", "min_size": 2, "max_size": 3},
            ]
        )
    )
    solution_fails: DataFrame = DataFrame(
        [
            {"student": "alice", "course": "c1"},
            {"student": "bob", "course": "c1"},
            {"student": "charlie", "course": "c1"},
            {"student": "dan", "course": "c1"},
        ]
    )
    solution_passes: DataFrame = DataFrame(
        [
            {"student": "alice", "course": "c1"},
            {"student": "bob", "course": "c1"},
            {"student": "charlie", "course": "c1"},
        ]
    )
    assert not all_courses_respect_max_nr_students(solution_fails, courses)
    assert all_courses_respect_max_nr_students(solution_passes, courses)


def all_students_assigned_to_a_preferred_course(
    solution: DataFrame, students: StudentsRegistry
):
    student_id: int
    preferred_courses: str
    student: Student
    assignment_by_student: pandas.Series = solution.set_index("student")["course"]
    for student_id, student in students.all_student_ids_and_students():
        preferred_courses: List[str] = student.preferences
        assigned_course: str = assignment_by_student[student_id]
        if assigned_course not in preferred_courses:
            return False
    return True


def test_all_students_assigned_to_a_preferred_course():
    students: StudentsRegistry = StudentsRegistry([Student(preferences=["gardening"])])
    solution_fails: DataFrame = DataFrame(
        [{"student": 0, "course": "kite surfing"},]
    )
    solution_passes: DataFrame = DataFrame(
        [{"student": 0, "course": "gardening"},]
    )
    assert not all_students_assigned_to_a_preferred_course(solution_fails, students)
    assert all_students_assigned_to_a_preferred_course(solution_passes, students)


def test_solves_problem():
    students: StudentsRegistry = StudentsRegistry(
        [Student(preferences=["course_1"]), Student(preferences=["course_2"]),]
    )
    courses: CourseRegistry = CourseRegistry(
        DataFrame(
            [
                {"name": "course_1", "min_size": 0, "max_size": 1},
                {"name": "course_2", "min_size": 0, "max_size": 1},
            ]
        )
    )
    solution: DataFrame = sorting_hat.solve(students, courses)
    assert solution is not None
    assert all_courses_respect_min_nr_students(solution, courses)
    assert all_courses_respect_max_nr_students(solution, courses)
    assert all_students_assigned_to_a_preferred_course(solution, students)


def test_solves_from_file():
    capacity_path: Path = Path("course_capacity.csv")
    student_path: Path = Path("student_preferences.csv")
    solution_path: Path = Path("test_solution.csv")
    solution_path.unlink(missing_ok=True)
    sorting_hat.solve_from_and_to_files(
        capacity_path, student_path, solution_path, encoding=None
    )
    assert solution_path.is_file()


def test_makes_courses_from_dataframe(course_info):
    courses: CourseRegistry = CourseRegistry(course_info)
    assert len(courses) == 1


def test_returns_course_names(course_info):
    courses: CourseRegistry = CourseRegistry(course_info)
    assert courses.get_all_course_names() == ["course_1"]


@pytest.fixture()
def courses(course_info: DataFrame) -> CourseRegistry:
    return CourseRegistry(course_info)


def test_gets_max_students_by_course_name(courses):
    assert courses.get_max_students_by_course_name("course_1") == 1


def test_raises_exception_when_making_courses_with_incorrect_field_names():
    with pytest.raises(ValueError):
        _ = CourseRegistry(DataFrame([{"non_existent_field": 1}]))


def test_gets_min_nr_students_by_course_name(courses: CourseRegistry):
    result: int = courses.get_min_students_by_course_name("course_1")
    assert result == 0


def test_can_get_students_ids():
    students: StudentsRegistry = StudentsRegistry(
        [Student(preferences=["c1"]), Student(preferences=["c1"]),]
    )
    ids: List[int] = students.all_student_registry_ids()

    expected: List[int] = [0, 1]
    assert ids == expected


def test_can_create_new_student():
    student: Student = Student(preferences=["course_1"])
    assert student is not None


def test_student_registry_has_length():
    registry: StudentsRegistry = StudentsRegistry([Student(preferences=["course_1"])])
    assert len(registry)


def test_registry_can_produce_list_of_student_ids_and_preferences():
    students: List[Student] = [Student(preferences=["course_1"])]
    registry = StudentsRegistry(students)
    student_ids_and_preferences = list(registry.all_student_ids_and_students())
    assert len(student_ids_and_preferences) == len(students)
    for student_id, student in student_ids_and_preferences:
        assert student_id is not None
        assert isinstance(student, Student)


def test_registry_gets_student_by_id():
    students: List[Student] = [Student(preferences=["course_1"])]
    registry = StudentsRegistry(students)
    actual: Student = registry.student_by_id(0)
    expected: Student = students[0]
    assert actual == expected
